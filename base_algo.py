from alfred.utils.recorder import TrainingIterator
from utils.ml import *

import torch


class BaseAlgo(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def evaluate_perf(self, env, n_episodes, seed, sample=False, render=False):
        if seed is not None:
            temp_seed = np.random.randint(0, 2 ** 16 - 1)
            set_seeds(env=env, seed=seed)
        Ti = TrainingIterator(n_episodes)
        for it in Ti:
            ret = 0
            obs = env.reset()
            done = False
            while not done:
                if render and it.itr == 0:
                    env.render()
                action = self.act(obs=obs, sample=sample)
                next_obs, reward, done, _ = env.step(action)
                obs = next_obs
                ret += reward
            it.record('eval_return', ret)
        if seed is not None:
            set_seeds(env=env, seed=temp_seed)
        return Ti.pop_all_means()

    def save(self, filepath):
        save_dict = {'init_dict': self.init_dict, 'params': self.get_params()}
        save_checkpoint(save_dict, filename=filepath)

    @classmethod
    def init_from_save(cls, filename, device):
        save_dict = torch.load(filename, map_location=device)
        return cls.init_from_dict(save_dict)

    @classmethod
    def init_from_dict(cls, save_dict):
        instance = cls(**save_dict['init_dict'])
        instance.load_params(save_dict['params'])
        return instance

    def load_params(self, save_dict):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def prep_rollout(self, rollout_device):
        raise NotImplementedError

    def prep_training(self, train_device):
        raise NotImplementedError

    def act(self, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

    def save_training_graphs(self, *args, **kwargs):
        raise NotImplementedError

    def wandb_watchable(self):
        raise NotImplementedError
