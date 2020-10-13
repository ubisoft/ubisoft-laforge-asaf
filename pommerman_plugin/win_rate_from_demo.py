from utils.data_structures import load_expert_demos
from pommerman.constants import Result
from alg_task_lists import POMMERMAN_UNWRAPPED_TASKS

import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, choices=POMMERMAN_UNWRAPPED_TASKS, required=True)
    parser.add_argument('--demos_name', type=str)
    parser.add_argument('--track_the_losses_of', type=int, nargs='+', default=None,
                        help="If the game is won but none of the winners is in this list then "
                             "the game is considered as lost")
    return parser.parse_args()


def check_win_rates(args):
    demos = load_expert_demos(args.task_name, args.demos_name)
    infos = demos['infos']
    n_demos = len(infos)
    n_agents = len(demos['trajectories'][0][0][0])
    wins = np.zeros(n_agents)
    n_ties = 0
    lost_games = []
    tie_games = []
    for i, info_traj in enumerate(infos):
        end_results = info_traj[-1]
        if end_results['result'] == Result.Win:
            winners = end_results['winners']
            wins[winners] += 1
            if args.track_the_losses_of is not None:
                if not set(winners).issubset(set(args.track_the_losses_of)):
                    lost_games.append(i)
        elif end_results['result'] == Result.Tie:
            n_ties += 1
            tie_games.append(i)

    win_rates = wins / n_demos
    tie_rates = n_ties / n_demos

    print(f'Win rates : {win_rates}  Tie rates : {tie_rates}\n')
    print(f'Games lost by {args.track_the_losses_of} : {lost_games}   Tied games: {tie_games}')


if __name__ == '__main__':
    args = get_args()
    check_win_rates(args)
