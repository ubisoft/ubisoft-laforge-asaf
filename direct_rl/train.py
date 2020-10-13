from alfred.utils.recorder import Recorder, TrainingIterator
from alfred.utils.misc import create_management_objects, keep_two_signif_digits
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json, load_dict_from_json, save_dict_to_json

from algo_manager import overide_args, init_from_config, should_update_rl
from env_manager import make_env
from utils.ml import save, remove, set_seeds
from utils.misc import get_computing_devices
from utils.data_structures import *
import utils.ml as ml
from pommerman_plugin.misc import load_game_states_from_demos
from alg_task_lists import RL_ALGS, RL_ON_POLICY_ALGS, RL_OFF_POLICY_ALGS, POMMERMAN_TASKS, TASKS

import os
import torch
import logging
from pathlib import Path
from collections import deque
from nop import NOP
import argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')


def get_training_args(overwritten_args=None):
    parser = argparse.ArgumentParser(description='PyTorch RL')
    parser.add_argument('--overide_default_args', type=parse_bool, default=False)

    # Alfred's arguments

    parser.add_argument('--alg_name', type=str, default='', choices=RL_ALGS,
                        help='name of the algorithm')
    parser.add_argument('--task_name', type=str, default='', choices=TASKS,
                        help='name of the task/task_name')
    parser.add_argument('--desc', type=str, default='',
                        help='description of the experiment to be run')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    # Algo-agnostic hyper-params

    parser.add_argument('--gamma', type=float, default=0.99,
                        help="MDP discount factor")
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='maximal number of collected episodes during whole run')
    parser.add_argument('--max_transitions', type=int, default=None,
                        help='maximal number of collected transitions during whole run')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='hidden size of the networks')
    parser.add_argument('--validation_seed', type=int, default=None)

    # Shared hyper-params

    parser.add_argument('--batch_size', type=int, default=-1,
                        help='batch size in transitions for RL update')
    parser.add_argument('--lr', type=float, default=-1,
                        help='learning rate for RL update')
    parser.add_argument('--grad_norm_clip', type=float, default=1e7,
                        help='grad norm clipping threshold')

    # PPO hyper-params

    parser.add_argument('--lamda', type=float, default=-1,
                        help='PPO GAE hyper-parameter')
    parser.add_argument('--episodes_between_updates', type=int, default=None,
                        help='number of EPISODES to collect before learning update')
    parser.add_argument('--epochs_per_update', type=int, default=-1,
                        help='number of passes through the collected data')
    parser.add_argument('--update_clip_param', type=float, default=0.2,
                        help='Update clipping parameter for PPO (default: 0.2)')
    parser.add_argument('--critic_lr_coef', type=float, default=1.,
                        help='critic_lr = critic_lr_coef * lr')

    # SAC hyper-params

    parser.add_argument('--transitions_between_updates', type=int, default=None,
                        help='number of TRANSITIONS to collect before learning update (transitions)')
    parser.add_argument('--replay_buffer_length', type=int, default=1000000,
                        help='max number of stored transitions (only for off-policy)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='target networks update rate')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='number of TRANSITIONS to collect before learning starts')
    parser.add_argument('--init_alpha', type=float, default=0.4,
                        help='initial entropy weight alpha')

    # SQIL hyper-params

    parser.add_argument('--demos_name', type=str, default='expert_demo_25.pkl',
                        help='demonstration file\'s name to initialize SQIL\'s')
    parser.add_argument('--demos_folder', type=str, default=None,
                        help='If None uses task_name as demos_folder')

    # Management config

    parser.add_argument('--episodes_between_saves', type=int, default=None)
    parser.add_argument('--transitions_between_saves', type=int, default=None)
    parser.add_argument('--number_of_eval_episodes', type=int, default=10)
    parser.add_argument('--sample_in_eval', type=parse_bool, default=False,
                        help='if true, uses the stochastic policy at evaluation, '
                             'otherwise uses the deterministic greedy policy')
    parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)
    parser.add_argument('--render', type=parse_bool, default=False,
                        help='render one episode for each RL loop')
    parser.add_argument('--render_evaluate', type=parse_bool, default=False,
                        help='renders one episode of each evaluate performance step')
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--wandb_watch_models', type=parse_bool, default=False,
                        help='track and sync model\'s parameters and gradients')
    parser.add_argument('--save_best_model_to_wandb', type=parse_bool, default=False)
    parser.add_argument('--sync_wandb', type=parse_bool, default=False)
    parser.add_argument('--use_wandb', type=parse_bool, default=False)
    parser.add_argument('--use_gpu', type=parse_bool, default=True)
    parser.add_argument('--do_rollouts_on_cpu', type=parse_bool, default=False)

    return parser.parse_args(overwritten_args)


def sanity_check_args(config, logger, alg_name):
    old_dict = config.__dict__.copy()
    if config.overide_default_args:
        logger.warning("\n You are overiding default config ! \n")

    # Sanity demonstrations path
    if config.demos_folder is None and config.task_name in POMMERMAN_TASKS:
        config.demos_folder = config.task_name.replace('learnable', 'agent47')

    # Sanity check for algorithms configs

    if alg_name in RL_OFF_POLICY_ALGS:
        if not (config.epochs_per_update == -1):
            logger.warning(
                f"For Off-Policy algo {alg_name} argument epochs_per_update has no effect. It is thus\
                 updated from {config.epochs_per_update} to -1")
            config.epochs_per_update = -1

    if config.alg_name in ['sqil', 'sqil-c']:

        if config.demos_folder is None:
            config.demos_folder = config.task_name

        demo_trajectories = load_expert_trajectories(task_name=config.task_name,
                                                     demos_name=config.demos_name,
                                                     demos_folder=config.demos_folder)

        # Gets state space ranges for env wrappers

        expert_traj_collection = TrajCollection([Traj(traj=demo) for demo in demo_trajectories])
        n_expert_transitions = expert_traj_collection.n_transitions
        half_batch_size = config.batch_size // 2
        if half_batch_size > n_expert_transitions:
            logger.warning(f'The batch_size cannot be lager than twice the expert data therefore it is'
                           f'automatically set to {2 * n_expert_transitions}')
            config.batch_size = 2 * n_expert_transitions

    # Either `max_transitions` or `max_episodes` should be provided. (One of them should be `None`.)
    if (config.max_transitions is None and config.max_episodes is None) or \
            (config.max_transitions is not None and config.max_episodes is not None):
        raise AttributeError("One and only one of `max_transitions` and `max_episodes` should be defined.")

    # Either `transitions_between_saves` or `max_episodes_between_saves` should be provided. (One of them should be `None`.)
    if (config.transitions_between_saves is None and config.episodes_between_saves is None) or \
            (config.transitions_between_saves is not None and config.episodes_between_saves is not None):
        raise AttributeError("One and only one of `transitions_between_saves` and `episodes_between_saves` should be defined.")

    if (config.max_transitions is None and config.transitions_between_saves is not None) or \
            (config.max_episodes is None and config.episodes_between_saves is not None):
        raise AttributeError("Logging should be matched with maximum iterations.")

    # if we modified the config we redo a sanity check
    if old_dict != config.__dict__:
        config = sanity_check_args(config, logger, alg_name)

    return config


def train(config, dir_tree=None, pbar="default_pbar", logger=None):
    # Overide config

    if config.overide_default_args:
        overide_args(args=config, alg_name=config.alg_name)

    dir_tree, logger, pbar = create_management_objects(dir_tree=dir_tree, logger=logger, pbar=pbar, config=config)
    if pbar is not None:
        pbar.total = config.max_transitions if config.max_episodes is None else config.max_episodes

    # Manages GPU usage

    train_device, rollout_device, logger = get_computing_devices(use_gpu=config.use_gpu,
                                                                 torch=torch,
                                                                 do_rollouts_on_cpu=config.do_rollouts_on_cpu,
                                                                 logger=logger)
    config.train_device = train_device.type
    config.rollout_device = rollout_device.type

    # Sanity check config and save it

    config = sanity_check_args(config=config, logger=logger, alg_name=config.alg_name)
    config.experiment_name = str(dir_tree.experiment_dir)
    save_config_to_json(config, str(dir_tree.seed_dir / 'config.json'))

    if (dir_tree.seed_dir / "config_unique.json").exists():
        config_unique_dict = load_dict_from_json(dir_tree.seed_dir / "config_unique.json")
        config_unique_dict.update((k, config.__dict__[k]) for k in config_unique_dict.keys() & config.__dict__.keys())
        save_dict_to_json(config_unique_dict, str(dir_tree.seed_dir / 'config_unique.json'))

    # Importing wandb (or not)

    if not config.sync_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'
        os.environ['WANDB_DISABLE_CODE'] = 'true'

    if config.use_wandb:
        import wandb
        os.environ["WANDB_DIR"] = str(dir_tree.seed_dir.absolute())
        wandb.init(id=dir_tree.get_run_name(), project='il_without_rl', entity='irl_la_forge', reinit=True)
        wandb.config.update(config, allow_val_change=True)
        wandb_save_dir = Path(wandb.run.dir) if config.save_best_model_to_wandb else None
    else:
        wandb = NOP()
        wandb_save_dir = None

    # Create env

    env = make_env(config.task_name)
    test_env = make_env(config.task_name)
    set_seeds(config.seed, env)
    set_seeds(config.seed, test_env)
    env_dims = ml.get_env_dims(env)

    # Create learner

    learner = init_from_config(env_dims, config)

    # Creates Experience Buffer

    if config.alg_name in RL_OFF_POLICY_ALGS:
        if config.alg_name in ['sqil', 'sqil-c']:
            demo_trajectories = load_expert_trajectories(task_name=config.task_name,
                                                         demos_name=config.demos_name,
                                                         demos_folder=config.demos_folder)

            # Put expert demonstrations into buffer

            buffer = SQILBuffer(demo_trajectories=demo_trajectories,
                                max_transitions=config.replay_buffer_length,
                                obs_space=env_dims['obs_space'],
                                act_space=env_dims['act_space'])

            logger.info(f"Number of expert trajectories: {buffer.n_demonstration_trajectories} \n"
                        f"Number of expert transitions: {buffer.n_demonstration_transitions} ")

            # Loading init_states from demos

            if config.task_name in POMMERMAN_TASKS:
                demos = load_expert_demos(config.demos_folder, config.demos_name)
                env.init_game_states = load_game_states_from_demos(demos, idx=0)
                test_env.init_game_states = env.init_game_states

        else:
            buffer = ReplayBuffer(max_transitions=config.replay_buffer_length,
                                  obs_space=env_dims['obs_space'],
                                  act_space=env_dims['act_space'])

    elif config.alg_name in RL_ON_POLICY_ALGS:
        buffer = OnPolicyBuffer(obs_space=env_dims['obs_space'])

    else:
        raise NotImplementedError

    # Creates recorders

    if config.task_name in POMMERMAN_TASKS:
        learner.metrics_to_record.add('n_woods')
        learner.metrics_to_record.add('n_enemies')

    os.makedirs(dir_tree.recorders_dir, exist_ok=True)
    train_recorder = Recorder(metrics_to_record=learner.metrics_to_record)

    # Initialize counters

    total_transitions = 0
    total_transitions_at_last_done = 0
    episode = 0
    epoch = 0
    cumul_eval_return = 0.
    eval_step = 0
    ret = 0
    best_eval_return = -float('inf')

    if config.max_episodes is not None:
        max_itr, heartbeat_ite = config.max_episodes, config.episodes_between_saves
    else:
        max_itr, heartbeat_ite = config.max_transitions, config.transitions_between_saves
    it = TrainingIterator(max_itr=max_itr, heartbeat_ite=heartbeat_ite)

    to_save = [learner]
    to_evaluate = learner
    to_watch = to_save if config.wandb_watch_models else []

    eval_perf_queue = deque(maxlen=5)
    ml.wandb_watch(wandb, to_watch)

    # Saves initial model

    ite_model_save_name = f'model_ep{episode}.pt'
    save(to_save, dir_tree.seed_dir, suffix=ite_model_save_name)

    # Initial reset

    state = env.reset()

    disp = config.render
    learner.prep_rollout(rollout_device)

    # Training loop

    while True:

        # ENVIRONMENT STEP

        if disp:
            env.render()

        if total_transitions <= config.warmup and config.alg_name in RL_OFF_POLICY_ALGS:
            action = learner.act(state, sample=True)
        else:
            action = learner.act(state, sample=True)
        next_state, reward, done, info = env.step(action)

        buffer.push(state, action, next_state, reward, ml.mask(done))

        state = next_state

        ret += reward
        total_transitions += 1
        if config.max_transitions is not None:
            it.touch()
            if it.itr <= config.warmup and config.alg_name in RL_OFF_POLICY_ALGS:
                it._heartbeat = False
            if pbar is not None:
                pbar.update()

        # episode ending

        if done:
            it.record('return', ret)
            it.record('episode_len', total_transitions - total_transitions_at_last_done)

            for info_key, info_value in info.items():
                if info_key in learner.metrics_to_record:
                    it.record(info_key, info_value)

            state = env.reset()

            ret = 0
            disp = False
            total_transitions_at_last_done = total_transitions
            episode += 1
            if config.max_episodes is not None:
                it.touch()  # increment recorder
                if pbar is not None:
                    pbar.update()

        # TRAINING STEP

        if should_update_rl(episode=episode,
                            done=done,
                            n_transitions=buffer.n_transitions,
                            total_transitions=total_transitions,
                            config=config,
                            name=config.alg_name):

            # Sample transitions (off-policy algo) or just takes the freshly collected (on-policy)

            if config.alg_name in RL_OFF_POLICY_ALGS:
                # we sample a batch normally
                experiences = buffer.sample(config.batch_size)
            else:
                # we flush the buffer (on-policy learning)
                experiences = buffer.flush()

            # Train the model

            learner.prep_training(train_device)  # train mode
            return_dict = learner.train_model(experiences)
            it.update(return_dict)
            learner.prep_rollout(rollout_device)  # back to eval mode

            epoch += 1

        # PLOTTING AND RECORDING

        if it.heartbeat:

            # Recording some metrics

            new_recordings = {'total_transitions': total_transitions,
                              'epoch': epoch,
                              'episode': episode,
                              'eval_step': eval_step}

            performance_metrics = to_evaluate.evaluate_perf(env=test_env,
                                                            n_episodes=config.number_of_eval_episodes,
                                                            seed=config.validation_seed,
                                                            sample=config.sample_in_eval,
                                                            render=config.render_evaluate)

            eval_step += 1
            cumul_eval_return += performance_metrics['eval_return']

            performance_metrics['cumul_eval_return'] = cumul_eval_return
            performance_metrics['avg_eval_return'] = cumul_eval_return / eval_step

            new_recordings.update(performance_metrics)

            means_to_log = it.pop_all_means()
            new_recordings.update(means_to_log)
            train_recorder.write_to_tape(new_recordings)
            train_recorder.save(dir_tree.recorders_dir / 'train_recorder.pkl')

            wandb.log(new_recordings, step=eval_step)

            # Saving best model

            eval_perf_queue.append(performance_metrics['eval_return'])
            running_avg_perf = np.mean(eval_perf_queue)

            if running_avg_perf >= best_eval_return:
                new_best = running_avg_perf
                best_eval_return = running_avg_perf
                save(to_save, dir_tree.seed_dir, f'model_best.pt')
                save(to_save, wandb_save_dir, f'model_best.pt')

                logger.info(
                    f"Eval step {eval_step}: Saved new best model at {str(dir_tree.seed_dir / 'model_best.pt')} "
                    f"with average perfo of {new_best}")

            # Saving current model periodically (even if not best)

            if (it.itr % (10 * heartbeat_ite)) == 0:
                remove(models=to_save, directory=dir_tree.seed_dir, suffix=ite_model_save_name)
                ite_model_save_name = f"model_eval_step_{eval_step}.pt"
                save(to_save, dir_tree.seed_dir, ite_model_save_name)

                logger.info(
                    f"Eval step {eval_step}: Saved model {str(dir_tree.seed_dir / f'model_eval_step_{eval_step}.pt')} "
                    f"(avg perfo {running_avg_perf})")

            # Creating and saving plots
            try:
                learner.save_training_graphs(train_recorder, dir_tree.seed_dir)
            except ImportError:
                pass
            disp = config.render

        if config.max_episodes is not None:
            if episode > config.max_episodes:
                break

        if config.max_transitions is not None:
            if total_transitions > config.max_transitions:
                break

    # Saving last model

    save(to_save, dir_tree.seed_dir, f"model_eval_step_{eval_step}.pt")

    logger.info(f"Saved last model: model_eval_step_{eval_step}.pt")
    logger.info(f"{Path(os.path.abspath(__file__)).parent.name}/{__file__}")

    # finishes logging before exiting training script

    wandb.join()


if __name__ == '__main__':
    args = get_training_args()
    train(config=args)
