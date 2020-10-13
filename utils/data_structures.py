from env_manager import get_observation_extractor
from utils.obs_dict import ObsDict, torch_cat_obs, np_concat_obs
from alg_task_lists import TOY_TASKS, POMMERMAN_TASKS, POMMERMAN_UNWRAPPED_TASKS, MUJOCO_TASKS
from utils.ml import to_torch
from utils.obs_dict import obs_to_torch

import torch
import numpy as np
import data
import os.path as osp
import pickle

from gym import spaces


class Traj:
    def __init__(self, traj=None, obs_space=None):

        if obs_space is not None:
            # if information about obs_spaces is available we initialize the dict from it
            self.storage = {'obs': ObsDict({obs_name.strip('_size'): [] for obs_name in obs_space.keys()}),
                            'next_obs': ObsDict({obs_name.strip('_size'): [] for obs_name in obs_space.keys()}),
                            'action': [],
                            'reward': [],
                            'mask': []}

        elif traj is not None:
            obs, _, next_obs, _, _ = traj[0]
            self.storage = {'obs': ObsDict({obs_name: [] for obs_name in obs.keys()}),
                            'next_obs': ObsDict({obs_name: [] for obs_name in next_obs.keys()}),
                            'action': [],
                            'reward': [],
                            'mask': []}

        # if some data is given we store it

        if traj is not None:
            self.store(traj)

    def store(self, traj):
        for transition in traj:
            obs, action, next_obs, reward, mask = transition

            for obs_name in self.storage['obs'].keys():
                self.storage['obs'][obs_name].append(np.asarray(obs[obs_name], dtype=np.float32))
                self.storage['next_obs'][obs_name].append(np.asarray(next_obs[obs_name], dtype=np.float32))

            self.storage['action'].append(np.asarray(action, dtype=np.float32))
            self.storage['reward'].append(np.asarray(reward, dtype=np.float32))
            self.storage['mask'].append(np.asarray(mask, dtype=np.float32))

    @classmethod
    def init_from_dict(cls, traj_as_dict):
        new_instance = cls(None, None)
        new_instance.storage = traj_as_dict
        return new_instance

    @property
    def obs(self):
        return self.storage['obs']

    @property
    def action(self):
        return self.storage['action']

    @property
    def next_obs(self):
        return self.storage['next_obs']

    @property
    def reward(self):
        return self.storage['reward']

    @property
    def mask(self):
        return self.storage['mask']

    def __iter__(self):
        for i in range(len(self.action)):
            yield ObsDict({obs_name: obs_value[i] for obs_name, obs_value in self.obs.items()}), \
                  self.action[i], \
                  ObsDict({obs_name: obs_value[i] for obs_name, obs_value in self.next_obs.items()}), \
                  self.reward[i], \
                  self.mask[i]

    def __len__(self):
        len_list = [len(values) for values in self.storage.values()]
        common_len = set(len_list)
        assert len(common_len) == 1, "traj elements have different len"
        return common_len.pop()

    def to_torch(self, keys_to_return, device):

        torch_traj = {key: (to_torch(np.asarray(self.storage[key]), device=device) if ('obs' not in key)
                            else obs_to_torch(self.storage[key], device=device)) for key in
                      keys_to_return}
        return Traj.init_from_dict(traj_as_dict=torch_traj)


class TrajCollection:
    "Just a list of paths"

    def __init__(self, paths):
        self.trajs = paths
        self.n_transitions = sum([len(traj) for traj in paths])

    def clear(self):
        self.trajs = []
        self.n_transitions = 0

    def push_traj(self, traj):
        self.add(traj)

    def add(self, traj):
        self.trajs.append(traj)
        self.n_transitions += len(traj)

    def extend(self, trajs):
        self.trajs.extend(trajs)
        self.n_transitions += sum([len(traj) for traj in trajs])

    def get(self, keys, from_torch=False):
        data = []
        for key in keys:

            key_data = [traj.storage[key] for traj in self.trajs]

            if key in ['obs', 'next_obs']:
                if from_torch:
                    key_data = torch_cat_obs(key_data, dim=0)
                else:
                    key_data = np_concat_obs(key_data, axis=0)
            else:
                if from_torch:
                    key_data = torch.cat(key_data, dim=0)
                else:
                    key_data = np.concatenate(key_data, axis=0)

            data.append(key_data)
        return data

    def get_list_of_traj_len(self):
        return [len(traj) for traj in self.trajs]

    def to_torch(self, keys_to_copy, device):
        torch_trajs = [t.to_torch(keys_to_return=keys_to_copy, device=device) for t in self.trajs]
        return TrajCollection(torch_trajs)

    def __len__(self):
        return len(self.trajs)

    def pop(self, idx):
        return self.trajs.pop(idx)

    def __iter__(self):
        for p in self.trajs:
            yield p

    def __getitem__(self, item):
        return self.trajs[item]

    @staticmethod
    def sample_batch(*args, batch_size):
        """
        Sample a batch of size batch_size from data.
        """
        N = len(args[0])

        batch_idxs = np.random.choice(np.arange(N), size=batch_size, replace=batch_size > N)

        indexed_data = []
        for data in args:
            if type(data) is ObsDict:
                indexed_data.append(
                    ObsDict({obs_name: obs_value[batch_idxs] for obs_name, obs_value in data.items()}))
            elif type(data) in [np.ndarray, torch.Tensor]:
                indexed_data.append(data[batch_idxs])

        return indexed_data


class ReplayBuffer(object):
    """
    This buffer supports having several types of observations (e.g. image and vector)
    Everywhere, observations are assumed to be a dictionary of obs_name:obs_value
    """

    def __init__(self, max_transitions, obs_space, act_space):
        self.max_transitions = max_transitions

        # Initializes all buffers with zeros
        self.obs_buff = ObsDict({
            obs_name.strip('_size'): np.zeros((max_transitions, *obs_shape), dtype=np.float16)
            for obs_name, obs_shape in obs_space.items()
        })
        self.next_obs_buff = ObsDict({
            obs_name.strip('_size'): np.zeros((max_transitions, *obs_shape), dtype=np.float16)
            for obs_name, obs_shape in obs_space.items()
        })
        self.obs_is_dict = True

        if isinstance(act_space, spaces.Discrete):
            act_size = 1
        elif isinstance(act_space, spaces.MultiDiscrete):
            act_size = len(act_space.nvec)
        else:
            act_size = act_space.shape[0]

        self.ac_buff = np.zeros((max_transitions, act_size), dtype=np.float16)
        self.rew_buff = np.zeros(max_transitions, dtype=np.float16)
        self.mask_buff = np.zeros(max_transitions, dtype=np.float16)

        self.filled_i = 0  # index of first empty location in buffer (last index + 1 when full)
        self.curr_i = 0  # current index to write to (overwrite oldest data)

    def __len__(self):
        return self.filled_i

    @property
    def n_transitions(self):
        return len(self)

    def push(self, observation, action, next_observation, reward, mask):
        # Add one transitions to the replay buffer

        if self.obs_is_dict:
            for obs_name, obs_value in observation.items():
                self.obs_buff[obs_name][self.curr_i:self.curr_i + 1] = np.asarray(obs_value, dtype=np.float32)
            for obs_name, obs_value in next_observation.items():
                self.next_obs_buff[obs_name][self.curr_i:self.curr_i + 1] = np.asarray(obs_value, dtype=np.float32)
        else:
            self.obs_buff[self.curr_i:self.curr_i + 1] = observation
            self.next_obs_buff[self.curr_i:self.curr_i + 1] = next_observation

        self.ac_buff[self.curr_i:self.curr_i + 1] = action
        self.rew_buff[self.curr_i:self.curr_i + 1] = reward
        self.mask_buff[self.curr_i:self.curr_i + 1] = mask

        # Update the pointers
        self.curr_i = (self.curr_i + 1) % self.max_transitions
        self.filled_i = min(self.filled_i + 1, self.max_transitions)

    def push_traj(self, traj):
        for transition in traj:
            self.push(*transition)

    def push_traj_collection(self, traj_col, **kwargs):
        for traj in traj_col:
            self.push_traj(traj)

    def sample(self, N, normalize_rewards=False):
        """
        Draw N samples uniformly from the buffer (with replacement)
        If N == -1: returns the entire buffer up to where it is filled (without shuffling)
        If normalize_rewards == True: returns ( rewards - rewards.mean() / rewards.std() )
        """
        if N == -1:
            inds = np.arange(0, len(self))
        else:
            inds = np.random.randint(self.filled_i, size=(N,))

        if normalize_rewards:
            ret_rews = (self.rew_buff[inds] - self.rew_buff[:self.filled_i].mean()) \
                       / self.rew_buff[:self.filled_i].std()
        else:
            ret_rews = self.rew_buff[inds]

        if self.obs_is_dict:
            return [ObsDict({obs_name: self.obs_buff[obs_name][inds] for obs_name in self.obs_buff.keys()}),
                    self.ac_buff[inds],
                    ObsDict({obs_name: self.next_obs_buff[obs_name][inds] for obs_name in self.next_obs_buff.keys()}),
                    ret_rews,
                    self.mask_buff[inds]]
        else:
            return (
                self.obs_buff[inds],
                self.ac_buff[inds],
                self.next_obs_buff[inds],
                self.rew_buff[inds],
                self.mask_buff[inds]
            )


class SQILBuffer(object):
    """
    Mixed Replay Buffer for SQIL
    It contains both expert demonstrations (self.demonstrations_buffer) and transitions gathered by interacting
    with the environment (self.interaction_buffers). The transitions in self.demonstrations_buffer are all
    associated to a reward of 1 and the transitions in self.interactions_buffer are associated with a reward of 0.
    Sampling from this buffer returns a batch of shuffled transitions, 50-50 from demos and interactions.
    see: https://arxiv.org/abs/1905.11108
    """

    def __init__(self, demo_trajectories, max_transitions, obs_space, act_space):

        self.n_demonstration_trajectories = len(demo_trajectories)
        n_demo_transitions = sum([len(traj) for traj in demo_trajectories])

        # Instantiate demonstrations buffer

        self.demos_buffer = ReplayBuffer(max_transitions=n_demo_transitions, obs_space=obs_space, act_space=act_space)

        # Push expert demonstrations in demonstrations buffer (with reward=1.)

        for trajectory in demo_trajectories:
            for transition in trajectory:
                o, a, n_o, _, m = transition
                self.demos_buffer.push(observation=o, action=a, next_observation=n_o, reward=1., mask=m)

        # Instantiate interactions buffer

        self.interact_buffer = ReplayBuffer(max_transitions=max_transitions - n_demo_transitions, obs_space=obs_space,
                                            act_space=act_space)

    def __len__(self):
        return self.n_interaction_transitions

    @property
    def n_transitions(self):
        return self.n_interaction_transitions

    @property
    def n_demonstration_transitions(self):
        return self.demos_buffer.filled_i

    @property
    def n_interaction_transitions(self):
        return self.interact_buffer.filled_i

    def push(self, observation, action, next_observation, reward, mask):
        """
        Adds collected transition in the buffer
        Reward is set to 0 for transitions collected by interacting with the environment
        """
        self.interact_buffer.push(observation, action, next_observation, reward=0., mask=mask)

    def sample(self, N, normalize_rewards=False):
        """
        Samples half of the transitions in each buffer, merge and shuffle them
        """
        d_obs, d_act, d_next, d_rew, d_mask = self.demos_buffer.sample(N // 2, normalize_rewards)
        i_obs, i_act, i_next, i_rew, i_mask = self.interact_buffer.sample(N // 2, normalize_rewards)

        idxs = np.random.permutation(len(d_act) + len(i_act))

        obs = ObsDict({obs_name: np.concatenate([d_obs[obs_name], i_obs[obs_name]])[idxs]
                       for obs_name in d_obs.keys()})

        next = ObsDict({obs_name: np.concatenate([d_next[obs_name], i_next[obs_name]])[idxs]
                        for obs_name in d_obs.keys()})

        act = np.concatenate([d_act, i_act])[idxs]
        rew = np.concatenate([d_rew, i_rew])[idxs]
        mask = np.concatenate([d_mask, i_mask])[idxs]

        return obs, act, next, rew, mask


class OnPolicyBuffer(object):
    """
    Variable length Replay Buffer
    This buffer supports having several types of observations (e.g. image and vector)
    Everywhere, observations are assumed to be a dictionary of obs_name:obs_value
    """

    def __init__(self, obs_space):
        self.obs_space = obs_space

        self.obs_buff = ObsDict({obs_name.strip('_size'): [] for obs_name in obs_space.keys()})
        self.next_obs_buff = ObsDict({obs_name.strip('_size'): [] for obs_name in obs_space.keys()})
        self.ac_buff = []
        self.rew_buff = []
        self.mask_buff = []

    def push(self, observation, action, next_observation, reward, mask):
        # Add one transitions to the replay buffer
        for obs_name, obs_value in observation.items():
            self.obs_buff[obs_name].append(np.asarray(obs_value, dtype=np.float32))

        for obs_name, obs_value in next_observation.items():
            self.next_obs_buff[obs_name].append(np.asarray(obs_value, dtype=np.float32))

        self.ac_buff.append(np.asarray(action, dtype=np.float32))
        self.rew_buff.append(np.asarray(reward, dtype=np.float32))
        self.mask_buff.append(mask)

    def __len__(self):
        return len(self.mask_buff)

    @property
    def n_transitions(self):
        return len(self)

    def get_all_current_as_np(self):
        obs = ObsDict({obs_name: np.asarray(self.obs_buff[obs_name])
                       for obs_name in self.obs_buff.keys()})

        next_obs = ObsDict({obs_name: np.asarray(self.next_obs_buff[obs_name])
                            for obs_name in self.next_obs_buff.keys()})

        ac = np.asarray(self.ac_buff)
        rew = np.asarray(self.rew_buff)
        mask = np.asarray(self.mask_buff)
        return obs, ac, next_obs, rew, mask

    def clear(self):
        self.__init__(obs_space=self.obs_space)

    def flush(self):
        """
        Returns the content of the buffer and re-initialises it
        """
        data = self.get_all_current_as_np()

        self.clear()
        return data


def batch_iter(all_obs, all_act, batch_size):
    """
    Iterable that yields observations and actions in batches of batch_size
    """
    assert len(all_obs[list(all_obs.keys())[0]]) == len(all_act)
    max_len = len(all_act)
    curr_i = 0
    while curr_i < max_len:
        go_to_i = min(max_len, curr_i + batch_size)
        obs = ObsDict({obs_name: obs_value[curr_i:go_to_i] for obs_name, obs_value in all_obs.items()})
        act = all_act[curr_i:go_to_i]
        curr_i = curr_i + go_to_i
        yield obs, act


def load_expert_demos(task_name, demos_name):
    folder = task_name
    demo_path = osp.join(osp.dirname(data.__file__), folder, demos_name)
    with open(demo_path, 'rb') as f:
        demos = pickle.load(f)
    return demos


def load_expert_trajectories(task_name, demos_name, demos_folder):
    demos = load_expert_demos(demos_folder, demos_name)

    if task_name in TOY_TASKS:
        assert demos_folder in TOY_TASKS, "For toy tasks, the demos must come from toy env"
        demo_trajectories = demos
        point_of_view = None
    elif task_name in POMMERMAN_TASKS:
        assert demos_folder in POMMERMAN_UNWRAPPED_TASKS, "For Pommerman tasks, the demos must come from Pommerman envs"
        demo_trajectories = demos['trajectories']

        # pommerman is a multi-agent environment so we must choose which agent we imitate
        if 'wins_from' in demos_name:
            point_of_view = int(demos_name.split('_')[-1].split('.')[0])
        else:
            point_of_view = 0
    elif task_name in MUJOCO_TASKS:
        assert demos_folder in MUJOCO_TASKS, "For MuJoCo tasks, the demos must come from MuJoCo env"
        demo_trajectories = demos
        point_of_view = None
    else:
        raise NotImplementedError

    obs_extractor = get_observation_extractor(task_name)

    for i, trajectory in enumerate(demo_trajectories):
        for j, transition in enumerate(trajectory):

            if point_of_view is not None:
                # we only select the transition for the agent of interest (mask is shared thus special treatement)
                transition = [item[point_of_view] for item in transition[:-1]] + [transition[-1]]

            o, a, n_o, r, m = transition
            if not (isinstance(o, ObsDict) and isinstance(n_o, ObsDict)):
                transition = (obs_extractor(o), a, obs_extractor(n_o), r, m)

            demo_trajectories[i][j] = transition

    return demo_trajectories
