from alfred.utils.config import parse_bool, parse_log_level
from alfred.utils.misc import create_logger

from alg_task_lists import TASKS, POMMERMAN_UNWRAPPED_TASKS
from env_manager import make_unwrapped_env, make_env
from pommerman_plugin.misc import dict_to_str, wait_for_ENTER_keypress
from pommerman_plugin.env_wrapper import PommermanWrapper
from pommerman_plugin.info_extractor import get_n_woods, get_n_enemies, get_n_powerUps

import argparse
import logging
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, choices=TASKS|POMMERMAN_UNWRAPPED_TASKS)
    parser.add_argument('--render', type=parse_bool, default=True)
    parser.add_argument('--n_episodes', type=int, default=1)
    parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)
    parser.add_argument('--waiting', type=parse_bool)
    return parser.parse_args()


def run_from_unwrapped(args, logger):
    env = make_unwrapped_env(task_name=args.task_name)

    all_returns = []
    all_n_transitions = []
    for i_episode in range(args.n_episodes):
        state = env.reset()
        done = False
        ret = None

        # transitions loop

        transition = 0
        while not done:

            transition += 1
            if args.render:
                env.render()

            # Sample actions for automated agents (all agents are automated in POMMERMAN_UNWRAPPED_TASKS)

            actions = env.act(state)

            # Print information for debugging

            logger.debug("____________________________________________________________")
            for i, s in enumerate(state):
                logger.debug(f'++++++++++++ AGENT {i} +++++++++++++++++\n')
                logger.debug(f"------------- STATE {i} ----------------\n")
                logger.debug(dict_to_str(s))
                logger.debug(f"------------- OBS {i} ----------------\n")
                obs = PommermanWrapper.extract_observation(s)
                logger.debug(dict_to_str(obs))
            logger.debug("____________________________________________________________")
            logger.debug(f'\nn_woods={get_n_woods(obs_map=obs["obs_map"])}'
                  f'\nn_enemies={get_n_enemies(obs_map=obs["obs_map"])}'
                  f'\nn_={get_n_powerUps(obs_map=obs["obs_map"])}')

            # Freeze on current frame until user-input to continue

            if args.waiting:
                wait_for_ENTER_keypress()

            # Environment transition

            state, reward, done, info = env.step(actions)

            # Accumulate return and bookeeping

            if ret is None:
                ret = np.array(reward)
            else:
                ret += np.array(reward)

            logger.debug(f"reward={reward}")

        logger.info(f'Episode {i_episode} finished: {ret}')
        all_returns.append(ret)
        all_n_transitions.append(transition)

    env.close()

    # Take some stats on all returns

    all_returns = np.array(all_returns).T

    for player_i in range(len(all_returns)):
        unique, counts = np.unique(all_returns[player_i], return_counts=True)
        outcomes_player_i_dict = dict(zip(unique, counts))
        str_player_i = f"player{player_i}:\n"
        for k in outcomes_player_i_dict.keys():
            if k == -1:
                str_player_i += f"n_losses = {outcomes_player_i_dict[k]}:\n"
            if k == 0:
                str_player_i += f"n_ties = {outcomes_player_i_dict[k]}:\n"
            if k == 1:
                str_player_i += f"n_wins = {outcomes_player_i_dict[k]}:\n"
        logger.info(str_player_i)


    logger.info(f"Mean(n_transitions) = {np.mean(all_n_transitions):.2f}")
    logger.info(f"Std(n_transitions) = {np.std(all_n_transitions):.2f}")
    logger.info(f"Max(n_transitions) = {np.max(all_n_transitions)}")
    logger.info(f"Min(n_transitions) = {np.min(all_n_transitions)}")


def run_from_wrapped(args, logger):
    env = make_env(task_name=args.task_name)
    # Run the episodes just like OpenAI Gym
    for i_episode in range(args.n_episodes):
        env.reset()
        done = False
        logger.debug(env.action_space)
        logger.debug(env.observation_space)
        while not done:
            if args.render:
                env.render()
            action = 0
            obs, reward, done, info = env.step(action)
            logger.debug(f"reward={reward}")
        logger.debug(f'Episode {i_episode} finished')
    env.close()


if __name__ == '__main__':
    args = get_args()
    logger = create_logger(name="run_pommerman", loglevel=args.log_level)
    if args.task_name in TASKS:
        run_from_wrapped(args, logger)
    else:
        run_from_unwrapped(args, logger)
