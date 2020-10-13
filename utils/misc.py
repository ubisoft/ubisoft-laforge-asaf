try:
    import pommerman
except ModuleNotFoundError:
    pass
try:
    import alfred.make_plot_arrays as make_plot_arrays
except ImportError:
    pass
from alfred.utils.directory_tree import DirectoryTree
from pathlib import Path


def get_computing_devices(use_gpu, torch, do_rollouts_on_cpu, logger):
    if use_gpu:
        if torch.cuda.is_available():
            train_device = torch.device('cuda:0')

        else:
            train_device = torch.device('cpu')
            logger.warning("You requested GPU usage but torch.cuda.is_available() returned False")
    else:
        train_device = torch.device('cpu')

    if do_rollouts_on_cpu:
        rollout_device = torch.device('cpu')
    else:
        rollout_device = train_device

    return train_device, rollout_device, logger


def uniquify(path):
    max_num = -1
    for file in path.parent.iterdir():
        if path.stem in file.stem:
            if path.suffix == file.suffix:
                num = str(file.stem).split('_')[-1]
                if not num == "":
                    num = int(num)
                    if num > max_num:
                        max_num = num
    if max_num == -1:
        return path.parent / (path.stem + f"_0" + path.suffix)
    else:
        return path.parent / (path.stem + f"_{max_num + 1}" + path.suffix)


def set_up_alfred():
    DirectoryTree.default_root = "./storage"
    DirectoryTree.git_repos_to_track['my_irl_framework'] = str(Path(__file__).parents[1])
    DirectoryTree.git_repos_to_track['pommerman'] = str(Path(pommerman.__file__).parents[1])
    try:
        make_plot_arrays.DEFAULT_PLOTS_TO_MAKE = [
            ('episode', 'eval_return', (None, None), (None, None)),
            ('episode', 'return', (None, None), (None, None)),
            ('episode', 'episode_len', (None, None), (0, 350)),
            ('episode', 'n_woods', (None, None), (0, 40)),  # for pommerman tasks
            ('episode', 'n_enemies', (None, None), (0, 4))  # for pommerman tasks
        ]
    except:
        pass
