
import pickle
import argparse
from pathlib import Path
from pommerman_plugin.make_expert import record_demonstrations
from alfred.utils.config import parse_bool


def extend_demos_args(overwritten_args=None):
    parser = argparse.ArgumentParser(description="extend_demos.py gives you the possibility to load some existing "
                                                 "demos and generate new ones so that the resulting larger set of "
                                                 "demos  have its first demos being the same as the loaded ones (it "
                                                 "basically creates a mother-set of demos from an existing one). "
                                                 "It gives you the possibility to use the same initial state as the "
                                                 "first loaded demo or to sample new ones for each new trajectory.")
    parser.add_argument("--expert_demos_path", type=str, required=True)
    parser.add_argument("--n_demos_to_add", type=int, required=True)
    parser.add_argument("--use_same_init_state_as_first_loaded_demo", type=parse_bool, default=True)
    return parser.parse_args(overwritten_args)


if __name__ == '__main__':
    args = extend_demos_args()

    # Load demos

    demos_path = Path(args.expert_demos_path)

    with open(str(demos_path), 'rb') as fp:
        loaded_demos = pickle.load(fp)

    # Extract demos info
    # WARNING: THIS IS SENSITIVE TO CHANGES IN NAMING CONVENTION
    # FEEL FREE TO RUN IN DEBUG TO MAKE SURE THIS IS WHAT YOU WANT!

    task_name = demos_path.parent.name
    n_demos_list = [args.n_demos_to_add]
    keep_only_wins_by = [int(id_str) for id_str in demos_path.stem.split('_')[1].strip('winsFrom').split('-')]
    n_different_init = int(demos_path.stem.split('_')[2].strip('nDifferentInit'))
    if args.use_same_init_state_as_first_loaded_demo:
        init_gstate = loaded_demos['game_states'][0][0]
    else:
        init_gstate = None

    new_demos, new_demos_suggested_path = record_demonstrations(task_name=task_name,
                                                                n_demos_list=n_demos_list,
                                                                keep_only_wins_by=keep_only_wins_by,
                                                                n_different_init=n_different_init,
                                                                init_gstate=init_gstate)

    # Merge both demos set

    mother_set = {key: loaded_demos[key] + new_demos[key] for key in new_demos.keys()}

    mother_set_path = demos_path.parent / f"expertDemo{len(mother_set['trajectories'])}_" \
                                          f'winsFrom{"-".join([str(id_int) for id_int in keep_only_wins_by])}_' \
                                          f'nDifferentInit{n_different_init}.pkl'

    with open(str(mother_set_path), 'wb') as fp:
        pickle.dump(mother_set, fp)
