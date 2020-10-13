import pickle
import argparse
from pathlib import Path
import random


def extract_demos_args(overwritten_args=None):
    parser = argparse.ArgumentParser(description="extract_demos.py gives you the possibility to load some existing "
                                                 "demos and save a new set which only contains the first n demos "
                                                 "(used to create a smaller set of expert demonstrations but that "
                                                 "contains the same trajectories as its mother-set). It also offer the "
                                                 "option to subsample the extracted demos.")
    parser.add_argument("--expert_demos_path", type=str, required=True)
    parser.add_argument("--n_demos_to_extract", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None, help="Keep only 1 out of <subsample> transitions")
    return parser.parse_args(overwritten_args)


if __name__ == '__main__':
    args = extract_demos_args()

    # Load demos

    demos_path = Path(args.expert_demos_path)

    with open(str(demos_path), 'rb') as fp:
        demos = pickle.load(fp)

    new_demos = {}
    for key in demos.keys():

        # Extact only first n_demos_to_extract

        if args.n_demos_to_extract is None:
            args.n_demos_to_extract = len(demos)
        else:
            assert len(demos[key]) > args.n_demos_to_extract, "you want to extract more demos than there are in original .pkl"

        new_demos[key] = [demos[key][i] for i in range(args.n_demos_to_extract)]

        # Subsample number of transitions in each demo

        if args.subsample is not None and key == "trajectories":
            subsampled_trajs = []
            for transitions in new_demos[key]:
                offset = random.randint(0, 20)
                subsampled_trajs.append([transitions[offset + i] for i in range(0, len(transitions) - offset, args.subsample)])

            new_demos[key] = subsampled_trajs

    new_demos_path = demos_path.parent / f"expert_demo_{args.n_demos_to_extract}_subsample1over{args.subsample}.pkl"

    with open(str(new_demos_path), 'wb') as fp:
        pickle.dump(new_demos, fp)
