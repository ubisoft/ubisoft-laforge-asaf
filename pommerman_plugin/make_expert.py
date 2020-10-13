from utils import ml
from pommerman.constants import Result
from pommerman_plugin.misc import get_game_state, reset_unwrapped_env_to_init_state
from env_manager import make_unwrapped_env
from alg_task_lists import POMMERMAN_UNWRAPPED_TASKS

import argparse
from pathlib import Path
import pickle
from tqdm import tqdm
import math
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, choices=POMMERMAN_UNWRAPPED_TASKS, required=True)
    parser.add_argument('--n_demos_list', default=[1, 5, 10, 20, 40, 80], type=int, nargs='+',
                        help="number of demonstrations in the sets")
    parser.add_argument('--keep_only_wins_by', default=[0], type=int, nargs='+',
                        help="Ids of the agents for which we will record the demonstration if one of them wins "
                             "the episode, if not specified we record all the episodes")
    parser.add_argument('--n_different_init', default=None, type=int,
                        help="Number of different initialisation states to record demonstration trajectories:"
                             "\n--n_different_init=None yields a different initialisation for each trajectory"
                             "\n--n_different_init=1 yields n_demos all starting in the same init state"
                             "\n--n_different_init=3 would yield n_demos/3 starting in each init state")
    return parser.parse_args()


def record_demonstrations(task_name, n_demos_list, keep_only_wins_by, n_different_init, init_gstate=None):
    path = Path('..') / 'data' / task_name
    path.mkdir(parents=True, exist_ok=True)
    env = make_unwrapped_env(task_name)
    pbar = tqdm(total=sum(n_demos_list))

    # we collect sets of demonstrations that have a varying number of demonstrations in it

    for n_demos in n_demos_list:
        demos_list = []
        infos_list = []
        game_states_list = []
        keep_only_wins_by = set(keep_only_wins_by) if keep_only_wins_by is not None else None

        if init_gstate is not None:
            n_different_init = 1
            new_init_interval = np.inf

        elif n_different_init is None or n_different_init >= n_demos:
            n_different_init = n_demos
            new_init_interval = 1

        else:
            n_different_init = n_different_init
            new_init_interval = int(math.ceil(n_demos / n_different_init))

        # we collect all the demonstrations for this set

        demo_i = 0
        while demo_i < n_demos:

            if init_gstate is not None:
                # use provided init_gstate
                state = reset_unwrapped_env_to_init_state(unwrapped_env=env, init_game_state=init_gstate)

            elif demo_i % new_init_interval == 0:
                # get new initialisation state
                state = reset_unwrapped_env_to_init_state(unwrapped_env=env, init_game_state=None)
                fixed_init_gstate = get_game_state(env)

            else:
                # re-use fixed initialisation state
                state = reset_unwrapped_env_to_init_state(unwrapped_env=env, init_game_state=fixed_init_gstate)

            # initialise containers

            gstate = get_game_state(env)
            transitions = []
            infos = []
            game_states = [gstate]

            # run an episode

            done = False
            while not done:
                actions = env.act(state)
                next_state, reward, done, info = env.step(actions)
                transitions.append([state, actions, next_state, reward, ml.mask(done)])
                infos.append(info)
                gstate = get_game_state(env)
                game_states.append(gstate)
                state = next_state

            # we record the demo only if the desired player wins or if we have no desired player

            if (keep_only_wins_by is None) \
                    or (
                    infos[-1]['result'] == Result.Win
                    and set(infos[-1]['winners']).intersection(keep_only_wins_by)):
                demos_list.append(transitions)
                infos_list.append(infos)
                game_states_list.append(game_states)
                pbar.update()
                demo_i += 1

        demonstrations = {'trajectories': demos_list, 'infos': infos_list, 'game_states': game_states_list}

        if keep_only_wins_by is None:
            suggested_path = path / f'expertDemo{n_demos}_' \
                                    f'nDifferentInit{n_different_init}.pkl'
        else:
            winners = "-".join([f"{w}" for w in sorted(list(keep_only_wins_by))])
            suggested_path = path / f'expertDemo{n_demos}_' \
                                    f'winsFrom{winners}_' \
                                    f'nDifferentInit{n_different_init}.pkl'

        return demonstrations, suggested_path


if __name__ == '__main__':
    args = get_args()
    demonstrations, suggested_path = record_demonstrations(task_name=args.task_name,
                                                           n_demos_list=args.n_demos_list,
                                                           keep_only_wins_by=args.keep_only_wins_by,
                                                           n_different_init=args.n_different_init,
                                                           init_gstate=None)

    with open(str(suggested_path), 'wb') as f:
        pickle.dump(demonstrations, f)

    print(f"Saved demos in {str(suggested_path)}")
