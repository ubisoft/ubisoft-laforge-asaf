from algo_manager import get_corresponding_train_script
from alg_task_lists import ALGS

# Setting up alfred
from utils.misc import set_up_alfred
set_up_alfred()

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name', type=str, choices=ALGS, required=True)

    main_args, extra_args = parser.parse_known_args()
    return main_args, extra_args


def main(config, dir_tree=None, logger=None, pbar="default_pbar"):

    # Get the corresponding training script and config based on specified algorithm

    train_script = get_corresponding_train_script(alg_names=[config.alg_name])

    # Runs the training script with provided arguments (and default config for unprovided options)

    train_script.train(config=config, dir_tree=dir_tree, logger=logger, pbar=pbar)


if __name__ == "__main__":
    # Parse commandline arguments
    # main_args: arguments defined in get_args()
    # extra_args: arguments passed to the user that are intended for get_training_args()
    #             of direct_rl/train.py or irl/train.py

    main_args, extra_args = get_args()

    # Get the corresponding training script and config based on specified algorithm

    train_script = get_corresponding_train_script(alg_names=[main_args.alg_name])
    train_args = train_script.get_training_args()

    main(config=train_args)
