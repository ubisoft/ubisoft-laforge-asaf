from alfred.utils.recorder import Recorder, TrainingIterator
from alfred.utils.misc import create_management_objects
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json, save_dict_to_json, load_dict_from_json

from utils.data_structures import *
from data.vizu_demo import get_expert_perfo
from utils.misc import get_computing_devices
from utils.ml import save, remove, set_seeds
import utils.ml as ml
import algo_manager as alg_manager
import direct_rl.train as drl_train_script
from alg_task_lists import IRL_ALGS, TASKS
from env_manager import make_env
from pommerman_plugin.misc import load_game_states_from_demos

import logging
import argparse
from pathlib import Path
import torch
import os
from collections import deque
from nop import NOP

import matplotlib

matplotlib.use('Agg')


def get_training_args(overwritten_args=None):
    parser = argparse.ArgumentParser(description='PyTorch IRL')
    parser.add_argument('--overide_default_args', type=parse_bool, default=False)

    # Alfred's arguments

    parser.add_argument('--alg_name', type=str, default='', choices=IRL_ALGS,
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
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='hidden size of the networks')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='maximal number of collected episodes during whole run')
    parser.add_argument('--max_transitions', type=int, default=None,
                        help='maximal number of collected transitions during whole run')
    parser.add_argument('--episodes_between_updates', type=int, default=None,
                        help='number of EPISODES to collect before learning update')
    parser.add_argument('--transitions_between_updates', type=int, default=None,
                        help='number of TRANSITIONS to collect before learning update')

    # Shared direct RL hyper-params

    parser.add_argument('--batch_size', type=int, default=-1,
                        help='batch size in transitions for RL update')
    parser.add_argument('--lr', type=float, default=-1,
                        help='learning rate for RL update')
    parser.add_argument('--grad_norm_clip', type=float, default=1e7,
                        help='grad norm clipping threshold for PPO/SAC')

    # PPO hyper-params

    parser.add_argument('--lamda', type=float, default=-1,
                        help='PPO GAE hyper-parameter')
    parser.add_argument('--epochs_per_update', type=int, default=-1,
                        help='number of passes through collected data')
    parser.add_argument('--update_clip_param', type=float, default=0.2,
                        help='clipping parameter for PPO (default: 0.2)')
    parser.add_argument('--critic_lr_coef', type=float, default=1.,
                        help='critic_lr = critic_lr_coef * lr')
    parser.add_argument('--critic_loss_coeff', type=float, default=0.5,
                        help='term multiplied to critic loss')

    # SAC hyper-params

    parser.add_argument("--replay_buffer_length", type=int, default=-1,
                        help='max number of stored transitions (only for off-policy)')
    parser.add_argument('--tau', type=float, default=-1,
                        help='target networks update rate')
    parser.add_argument('--warmup', type=int, default=-1,
                        help='number of TRANSITIONS to collect before learning starts')
    parser.add_argument('--init_alpha', type=float, default=-1, help='initial entropy weight alpha')

    # IRL hyper-params

    parser.add_argument('--demos_name', type=str, default='expert_demo_25.pkl',
                        help='demonstration file\'s name')
    parser.add_argument('--demos_folder', type=str, default=None,
                        help='if None uses task_name as demos_folder')
    parser.add_argument('--use_advantage_formulation', type=parse_bool, default=False,
                        help='whether or not the discriminator\'s model outputs a normalized log probability')
    parser.add_argument('--use_multi_head', type=parse_bool, default=True,
                        help='only for discrete action setting: one network head per action')
    parser.add_argument('--ent_wt', type=float, default=0.,
                        help='weight of the causal entropy regularization term in the learned RL reward (GAIL only)')
    parser.add_argument('--d_batch_size', type=int, default=None,
                        help='discriminator batch size in transitions or in trajectories depending on the algo')
    parser.add_argument('--d_epochs_per_update', type=int, default=None,
                        help='number of passes through the collected data for the discriminator')
    parser.add_argument('--d_lr', type=float, default=1e-3,
                        help='discriminator learning rate')
    parser.add_argument('--d_reward_definition', type=str, default='mixed', choices=['negative', 'positive', 'mixed'],
                        help='GAIL formulation of the reward from D, negative: log(D), positive: -log(1-D), '
                             'mixed: one of the above depending on the environment '
                             '(survival bonus or termination bonus)')
    parser.add_argument('--window_size', type=int, default=None,
                        help='for asaf-w, size of the sub-trajectories')
    parser.add_argument('--window_stride', type=int, default=None,
                        help='For asaf-w, stride of the sub-trajectories')
    parser.add_argument('--d_episodes_between_updates', type=int, default=None,
                        help='number of EPISODES to collect before learning update')
    parser.add_argument('--d_transitions_between_updates', type=int, default=None,
                        help='number of TRANSITIONS to collect before learning update')
    parser.add_argument('--gradient_penalty_coef', type=float, default=0.0,
                        help='gradient penalty coefficient for discriminator (GAIL only)')
    parser.add_argument('--d_grad_norm_clip', type=float, default=1e7,
                        help='gradient norm clipping threshold for discriminator')
    parser.add_argument('--d_grad_value_clip', type=float, default=1e7,
                        help='gradient value clipping threshold for discriminator')

    ## Management config

    parser.add_argument('--episodes_between_saves', type=int, default=None)
    parser.add_argument('--transitions_between_saves', type=int, default=None)
    parser.add_argument('--number_of_eval_episodes', type=int, default=10)
    parser.add_argument('--sample_in_eval', type=parse_bool, default=False,
                        help='if true, uses the stochastic policy at evaluation, '
                             'otherwise uses the deterministic greedy policy')
    parser.add_argument('--validation_seed', type=int, default=None)
    parser.add_argument('--heavy_record', type=parse_bool, default=False,
                        help='Record a lot of things')
    parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)
    parser.add_argument('--render', type=parse_bool, default=False,
                        help='renders one episode for each IRL loop')
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


def sanity_check_args(config, logger):
    old_dict = config.__dict__.copy()
    config = drl_train_script.sanity_check_args(config=config, logger=logger, alg_name=config.rl_alg_name)

    # Sanity check episode-wise v.s. transition-wise specification

    if config.episodes_between_updates is not None:
        assert config.d_episodes_between_updates is not None, "IL and RL should be based both either on transitions or on episodes"

    if config.transitions_between_updates is not None:
        assert config.d_transitions_between_updates is not None, "IL and RL should be based both either on transitions or on episodes"

    # Sanity check for algorithms configs

    if config.irl_alg_name in ["asqf", "airl"]:
        if config.irl_alg_name in ["asqf"]:
            config.use_multi_head = True
            config.use_advantage_formulation = False

        if config.irl_alg_name == "airl":
            config.use_multi_head = False
            config.use_advantage_formulation = False

    if config.irl_alg_name in ["asaf-full", "asaf-w", "asaf-1"]:
        if config.irl_alg_name == "asaf-full":
            config.break_traj_to_windows = False
            config.window_size = None
            config.window_stride = None
            config.window_over_episode = None

        elif config.irl_alg_name == "asaf-w":
            config.break_traj_to_windows = True
            if config.window_stride is None:
                config.window_stride = config.window_size
            else:
                assert config.window_stride <= config.window_size

            if config.d_episodes_between_updates is not None:
                config.window_over_episode = False
            elif config.d_transitions_between_updates is not None:
                config.window_over_episode = True

        elif config.irl_alg_name == "asaf-1":
            config.break_traj_to_windows = True
            config.window_size = 1
            config.window_stride = 1
            config.window_over_episode = True

    if config.irl_alg_name == "gail":
        if config.d_reward_definition not in ["positive", "negative"]:
            if config.task_name in ["mountaincar", "mountaincar-c"]:
                config.d_reward_definition = "negative"
            elif config.task_name in ["hopper-c", "walker2d-c", "halfcheetah-c", "ant-c", "humanoid-c"]:
                config.d_reward_definition = "positive"
            else:
                config.d_reward_definition = "positive"

    # Sanity checks for expert demonstrations

    if config.demos_folder is None and config.task_name in POMMERMAN_TASKS:
        config.demos_folder = config.task_name.replace('learnable', 'agent47')

    if config.demos_folder is None:
        config.demos_folder = config.task_name

    # if we modified the config we redo a sanity check
    if old_dict != config.__dict__:
        config = sanity_check_args(config, logger)

    return config


def train(config, dir_tree=None, pbar="default_pbar", logger=None):

    irl_alg_name, rl_alg_name = config.alg_name.split("X")
    config.irl_alg_name = irl_alg_name
    config.rl_alg_name = rl_alg_name

    # Overide config

    if config.overide_default_args:
        config = alg_manager.overide_args(args=config, alg_name=rl_alg_name)
        config = alg_manager.overide_args(args=config, alg_name=irl_alg_name)

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
    config = sanity_check_args(config=config, logger=logger)
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

    # Load demonstrations

    demo_trajectories = load_expert_trajectories(task_name=config.task_name,
                                                 demos_name=config.demos_name,
                                                 demos_folder=config.demos_folder)

    # Convert demos into TrajCollection

    expert_traj_collection = TrajCollection(
        [Traj(traj=demo, obs_space=env_dims['obs_space']) for demo in demo_trajectories])

    logger.info(f"Number of expert trajectories: {len(expert_traj_collection)} \n"
                f"Number of expert transitions: {expert_traj_collection.n_transitions} ")

    get_expert_perfo(env, expert_traj_collection, logger)

    # Loading init_states from Pommerman demos

    if config.task_name in POMMERMAN_TASKS:
        demos = load_expert_demos(config.demos_folder, config.demos_name)
        env.init_game_states = load_game_states_from_demos(demos, idx=0)
        test_env.init_game_states = env.init_game_states

    # Create learner

    irl_learner = alg_manager.named_init_from_config(env_dims, config, irl_alg_name)
    irl_learner.prep_training(train_device)
    irl_learner.add_expert_path(expert_traj_collection)
    irl_learner.prep_rollout(rollout_device)

    if rl_alg_name == '':
        rl_learner = irl_learner
    else:
        rl_learner = alg_manager.named_init_from_config(env_dims, config, rl_alg_name)

    # Creates buffers to store the data according to on-policy formulations

    buffer = OnPolicyBuffer(obs_space=env_dims['obs_space'])

    # Creates recorders

    if config.task_name in POMMERMAN_TASKS:
        irl_learner.metrics_to_record.add('n_woods')
        irl_learner.metrics_to_record.add('n_enemies')

    os.makedirs(dir_tree.recorders_dir, exist_ok=True)
    train_recorder = Recorder(metrics_to_record=irl_learner.metrics_to_record | rl_learner.metrics_to_record)

    # Initialize counters

    total_transitions = 0
    total_transitions_at_last_done = 0
    episode = 0
    irl_epoch = 0
    rl_epoch = 0
    cumul_eval_return = 0.
    eval_step = 0
    ret = 0
    best_eval_return = -float('inf')

    if config.max_episodes is not None:
        max_itr, heartbeat_ite = config.max_episodes, config.episodes_between_saves
    else:
        max_itr, heartbeat_ite = config.max_transitions, config.transitions_between_saves
    it = TrainingIterator(max_itr=max_itr, heartbeat_ite=heartbeat_ite)

    to_save = [irl_learner, rl_learner] if not rl_alg_name == '' else [irl_learner]
    to_evaluate = rl_learner  # the performance can only come from one model
    to_watch = to_save if config.wandb_watch_models else []

    eval_perf_queue = deque(maxlen=5)
    ml.wandb_watch(wandb, to_watch)

    # Saves initial model

    ite_model_save_name = f'model_ep{episode}.pt'
    save(to_save, dir_tree.seed_dir, suffix=ite_model_save_name)

    # Initial reset

    state = env.reset()

    disp = config.render
    rl_learner.prep_rollout(rollout_device)

    # Deals with on-policy formulation only for irl updates

    should_update_irl = alg_manager.should_update_on_policy_irl

    # Training loop

    while True:

        # ENVIRONMENT STEP

        if disp:
            env.render()

        action = rl_learner.act(state, sample=True)
        next_state, reward, done, info = env.step(action)

        buffer.push(state, action, next_state, reward, ml.mask(done))

        state = next_state

        ret += reward
        total_transitions += 1

        if config.max_transitions is not None:
            it.touch()
            if pbar is not None:
                pbar.update()

        # episode ending

        if done:
            it.record('return', ret)
            it.record('episode_len', total_transitions - total_transitions_at_last_done)

            for info_key, info_value in info.items():
                if info_key in irl_learner.metrics_to_record:
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

        # TRAIN DISCRIMINATOR

        if should_update_irl(episode=episode,
                             done=done,
                             n_transitions=buffer.n_transitions,
                             total_transitions=total_transitions,
                             config=config):

            # get the training data

            data = buffer.get_all_current_as_np()

            # we prep_rollout because we will eval the policy and not train it
            # but we set the device to train because data is on this device
            rl_learner.prep_rollout(train_device)
            irl_learner.prep_training(train_device)

            return_dict = irl_learner.fit(data=data,
                                          batch_size=config.d_batch_size,
                                          policy=rl_learner.get_policy(),
                                          n_epochs_per_update=config.d_epochs_per_update,
                                          logger=logger,
                                          heavy_record=config.heavy_record,
                                          config=config)

            # updates counters

            it.update(return_dict)
            irl_epoch += 1
            if rl_alg_name == "":
                # there is no explicit RL algo so we actually just updated the policy
                rl_epoch = irl_epoch

            # set models back to roll-out mode

            rl_learner.prep_rollout(rollout_device)
            irl_learner.prep_rollout(rollout_device)

            # handles the on-policy buffers

            if rl_alg_name == "":
                # the policy changed (because we updated generator) and thus we must flush the irl_buffer
                buffer.clear()

        # TRAIN RL

        if rl_alg_name != "" and alg_manager.should_update_rl(total_transitions=total_transitions,
                                                              episode=episode,
                                                              n_transitions=buffer.n_transitions,
                                                              done=done,
                                                              config=config,
                                                              name=rl_alg_name,
                                                              did_irl_update_first=irl_epoch > rl_epoch):

            # we prep_rollout because we will eval the discriminator and not train it
            # but we set the device to train because data is on this device
            irl_learner.prep_rollout(train_device)
            rl_learner.prep_training(train_device)

            # Take on-policy data and flushes buffer (because generator is about to change)

            experiences = buffer.flush()

            # Evaluate the estimated rewards

            experiences = irl_learner.update_reward(experiences=experiences,
                                                    policy=rl_learner.get_policy(),
                                                    ent_wt=config.ent_wt)

            # Train the model

            return_dict = rl_learner.train_model(experiences)
            it.update(return_dict)

            # Updates counter

            rl_epoch += 1

            # Set models back to roll-out mode

            rl_learner.prep_rollout(rollout_device)
            irl_learner.prep_rollout(rollout_device)

        # PLOTTING AND RECORDING

        if it.heartbeat:

            # Recording some metrics

            new_recordings = {'total_transitions': total_transitions,
                              'irl_epoch': irl_epoch,
                              'rl_epoch': rl_epoch,
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
                save(to_save, dir_tree.seed_dir, f'model_best.pt')
                save(to_save, wandb_save_dir, f'model_best.pt')
                best_eval_return = new_best

                logger.info(
                    f"Eval Step {eval_step}: Saved new best model at {str(dir_tree.seed_dir / 'model_best.pt')} "
                    f"with average perfo of {new_best}")

            # Saving current model periodically (even if not best)

            if (it.itr % (10 * it.heartbeat_ite)) == 0:
                remove(models=to_save, directory=dir_tree.seed_dir, suffix=ite_model_save_name)
                ite_model_save_name = f"model_eval_step_{eval_step}.pt"
                save(to_save, dir_tree.seed_dir, ite_model_save_name)

                logger.info(
                    f"Eval step {eval_step}: Saved model {str(dir_tree.seed_dir / f'model_ep_{eval_step}.pt')} "
                    f"(avg perfo {running_avg_perf})")

            # Creating and saving plots
            try:
                irl_learner.save_training_graphs(train_recorder, dir_tree.seed_dir)
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

    save(to_save, dir_tree.seed_dir, f"model_ep_{eval_step}.pt")

    logger.info(f"Saved last model: model_ep_{eval_step}.pt")
    logger.info(f"{Path(os.path.abspath(__file__)).parent.name}/{__file__}")

    # finishes logging before exiting training script

    wandb.join()


if __name__ == '__main__':
    args = get_training_args()
    train(config=args)
