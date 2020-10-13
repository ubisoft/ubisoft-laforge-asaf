from alfred.utils.directory_tree import DirectoryTree
from alfred.utils.recorder import TrainingIterator
from alfred.utils.config import parse_bool, load_config_from_json

from algo_manager import init_from_save
from utils.misc import uniquify
import utils.ml as ml
from utils.data_structures import load_expert_demos
from alg_task_lists import POMMERMAN_TASKS, MUJOCO_TASKS
from env_manager import make_env
from pommerman_plugin.misc import save_gif_from_png_folder, load_game_states_from_demos
from pommerman_plugin.misc import wait_for_ENTER_keypress
from mujoco_py.generated import const

import time
import pickle
import argparse
import imageio
from pathlib import Path
import torch


def get_evaluation_args(overwritten_args=None):
    parser = argparse.ArgumentParser()

    ## Shared hyper-params
    parser.add_argument('--storage_name', default='', type=str,
                        help='Name of the model storage')
    parser.add_argument('--root_dir', default=None, type=str)
    parser.add_argument('--experiment_num', type=int, default=1)
    parser.add_argument('--seed_num', default=1, type=int,
                        help='Seed directory in experiments folder')
    parser.add_argument('--eval_seed', default=1234567, type=int,
                        help='Random seed for evaluation rollouts')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--max_ep_len', default=1000000, type=int)
    parser.add_argument('--act_deterministic', type=parse_bool, default=True)
    parser.add_argument('--number_of_eps', type=int, default=3)
    parser.add_argument('--make_expert', type=parse_bool, default=False)
    parser.add_argument('--expert_save_path', type=str, default=None)
    parser.add_argument('--render', type=parse_bool, default=False)
    parser.add_argument('--make_gif', type=parse_bool, default=False)
    parser.add_argument('--n_skipped_frames', type=int, default=0)
    parser.add_argument('--fps', default=3000, type=int)
    parser.add_argument('--waiting', type=parse_bool, default=False)

    # In cases where the model was trained only from initial states from the demonstrations,
    # we can load these demonstrations and also evaluate on these same initial states

    parser.add_argument('--demos_name', type=str, default=None,
                        help='demonstration filename used to train the agent')
    parser.add_argument('--demos_folder', type=str, default=None,
                        help='demonstration folder in data/ used to train the agent')

    return parser.parse_args(overwritten_args)


def record_frame(env, iter, n_skipped_frames, task_name, frames, temp_png_folder):
    if iter % (n_skipped_frames + 1) == 0:
        if task_name in POMMERMAN_TASKS:
            env.render(record_pngs_dir=temp_png_folder)
        else:
            frames.append(env.render('rgb_array'))


def evaluate(args):
    # loads config and model

    dir_tree = DirectoryTree.init_from_branching_info(root_dir=args.root_dir, storage_name=args.storage_name,
                                                      experiment_num=args.experiment_num, seed_num=args.seed_num)

    config = load_config_from_json(dir_tree.seed_dir / "config.json")

    if args.model_name is not None:
        model_path = dir_tree.seed_dir / args.model_name
    else:
        if 'rl_alg_name' in config.__dict__.keys():
            if config.rl_alg_name == "":
                model_name = config.irl_alg_name
            else:
                model_name = config.rl_alg_name
        else:
            model_name = config.alg_name
        model_path = dir_tree.seed_dir / (model_name + '_model_best.pt')
    learner = init_from_save(model_path, device=torch.device('cpu'))

    if args.make_gif:
        gif_path = dir_tree.storage_dir / 'gifs'
        gif_path.mkdir(exist_ok=True)
        gif_full_name = uniquify(gif_path /
                                 f"{config.task_name}"
                                 f"_experiment{args.experiment_num}"
                                 f"_seed{args.seed_num}"
                                 f"_evalseed{args.eval_seed}.gif")

        if config.task_name in POMMERMAN_TASKS:
            temp_png_folder_base = uniquify(gif_path / 'temp_png')
        else:
            temp_png_folder = False

    # Makes task_name and recorders

    env = make_env(config.task_name)
    ml.set_seeds(args.eval_seed, env)
    Ti = TrainingIterator(args.number_of_eps)
    frames = []
    dt = 1. / args.fps
    trajectories = []

    # camera angles and stuff

    if config['task_name'] in MUJOCO_TASKS:
        env.render(mode='human' if args.render else 'rgb_array')
        env.unwrapped.viewer.cam.type = const.CAMERA_TRACKING

        # # Option 1 (FROM THE SIDE)
        # env.unwrapped.viewer.cam.trackbodyid = 0
        # env.unwrapped.viewer.cam.elevation = -25
        # env.unwrapped.viewer.cam.distance = 6

        # Option 2 (FROM PERSPECTIVE)
        env.unwrapped.viewer.cam.trackbodyid = 0
        env.unwrapped.viewer.cam.elevation = -15
        env.unwrapped.viewer.cam.distance = 4
        env.unwrapped.viewer.cam.azimuth = 35

    # Get expert demonstrations initial states

    if config.task_name in POMMERMAN_TASKS:

        if args.demos_folder is None:
            args.demos_folder = config.task_name.replace('learnable', 'agent47')

        demos = load_expert_demos(config.demos_folder, config.demos_name)
        env.init_game_states = load_game_states_from_demos(demos, idx=0)

    # Episodes loop

    for it in Ti:
        t = 0
        trajectory = []
        ret = 0
        done = False

        # Initial reset

        obs = env.reset()

        # Rendering options

        if args.make_gif:
            if config.task_name in POMMERMAN_TASKS:  # pommerman saves .png per episode
                temp_png_folder = temp_png_folder_base / f"ep_{it.itr}"
                temp_png_folder.mkdir(parents=True, exist_ok=True)
            record_frame(env, t, args.n_skipped_frames, config.task_name, frames, temp_png_folder)

        if args.render:
            env.render()

        if args.waiting:
            wait_for_ENTER_keypress()

        # transitions loop

        while not done:
            calc_start = time.time()
            action = learner.act(obs=obs, sample=not args.act_deterministic)
            next_obs, reward, done, _ = env.step(action)

            if args.make_expert:
                trajectory.append((obs, action, next_obs, reward, ml.mask(done)))

            obs = next_obs
            ret += reward
            t += 1

            if args.render:
                # Enforces the fps config
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                env.render('human')

            if args.waiting:
                wait_for_ENTER_keypress()

            if args.make_gif:
                # we want the last frame even if we skip some frames
                record_frame(env, t * (1 - done), args.n_skipped_frames, config.task_name, frames, temp_png_folder)
            if t > args.max_ep_len:
                break
        it.record('eval_return', ret)
        if args.make_expert:
            trajectories.append(trajectory)

    # Saves gif of all the episodes

    if args.make_gif:
        if config.task_name in POMMERMAN_TASKS:
            save_gif_from_png_folder(temp_png_folder_base, gif_full_name, 1 / dt, delete_folder=True)
        else:
            imageio.mimsave(str(gif_full_name), frames, duration=dt)
    env.close()

    # Saves expert_trajectories

    if args.make_expert:
        if args.expert_save_path is not None:
            expert_path = Path(args.expert_save_path)
        else:
            expert_path = Path('./data/' + config.task_name + f'/expert_demo_{args.number_of_eps}.pkl')

        expert_path.parent.mkdir(exist_ok=True, parents=True)
        expert_path = uniquify(expert_path)
        with open(str(expert_path), 'wb') as fp:
            pickle.dump(trajectories, fp)
            fp.close()
    return Ti.pop_all_means()['eval_return']


if __name__ == "__main__":
    args = get_evaluation_args()
    print(f'avg: {evaluate(args)}')
