from alfred.utils.config import parse_bool

from utils.misc import uniquify
from utils.data_structures import load_expert_demos
from pommerman_plugin.misc import wait_for_ENTER_keypress, save_gif_from_png_folder, reset_unwrapped_env_to_init_state
from env_manager import make_unwrapped_env
from alg_task_lists import POMMERMAN_UNWRAPPED_TASKS

import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, choices=POMMERMAN_UNWRAPPED_TASKS, required=True)
    parser.add_argument('--demos_name', type=str)
    parser.add_argument('--demos_indexes', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--waiting', type=parse_bool, default=False)
    parser.add_argument("--make_gif", type=parse_bool, default=False)
    parser.add_argument('--n_skipped_frames', type=int, default=0)
    parser.add_argument("--fps", type=float, default=5)
    return parser.parse_args()


def vizu_game_sequence(game_state_list, task_name, waiting=False, save_frames=False,
                       n_skipped_frames=0, temp_png_folder=None):
    env = make_unwrapped_env(task_name)

    for t, game_state in enumerate(game_state_list):
        reset_unwrapped_env_to_init_state(unwrapped_env=env, init_game_state=game_state)
        record_dir = None

        if save_frames:
            if (t % (n_skipped_frames + 1) == 0) or (t == (len(game_state_list) - 1)):
                record_dir = temp_png_folder

        env.render(record_pngs_dir=record_dir)
        if waiting:
            wait_for_ENTER_keypress()

    env.close()


def vizu_demos(args):
    demos = load_expert_demos(args.task_name, args.demos_name)
    game_states = demos['game_states']

    # creates gif path

    if args.make_gif:
        demos_path = Path('../data') / args.task_name / args.demos_name
        temp_png_folder_base = uniquify(demos_path.parent / 'temp_png')
        gif_full_path = demos_path.parent / (demos_path.stem + f"_demosIdxs{args.demos_indexes}_savedGif.gif")
        gif_full_path = uniquify(gif_full_path)
    else:
        gif_full_path = None

    # visualise demonstrations

    for dem_idx in args.demos_indexes:
        try:
            game_seq = game_states[dem_idx]
        except IndexError as e:
            print(f"Warning: dem_idx={dem_idx} is out of bound for demonstration set '{args.task_name}/{args.demos_name}'.")
            continue

        if args.make_gif:
            temp_png_folder = temp_png_folder_base / f'demo_{dem_idx}'
            temp_png_folder.mkdir(parents=True, exist_ok=True)
        else:
            temp_png_folder = None

        vizu_game_sequence(game_state_list=game_seq,
                           task_name=args.task_name,
                           waiting=args.waiting,
                           save_frames=args.make_gif,
                           n_skipped_frames=args.n_skipped_frames,
                           temp_png_folder=temp_png_folder)

    if args.make_gif:
        save_gif_from_png_folder(folder=temp_png_folder_base, gif_path=gif_full_path, fps=args.fps, delete_folder=True)


if __name__ == "__main__":
    args = get_args()
    vizu_demos(args)
