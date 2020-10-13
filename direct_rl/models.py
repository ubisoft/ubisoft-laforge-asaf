from utils.networks import MLPNetwork, CNN_MLP_hybrid
from utils.ml import gumbel_softmax, get_act_properties_from_act_space, to_torch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

MIN_LOG_STD = -20
MAX_LOG_STD = 2
EPS = 1e-8
EPS2 = 1e-4


################################### CONTINUOUS ##############################

class Q(nn.Module):
    
    def __init__(self, obs_space, act_space, hidden_size, lr, set_final_bias=False):
        super().__init__()

        act_size, is_discrete = get_act_properties_from_act_space(act_space)

        if set(obs_space.keys()) == {'obs_vec_size'}:
            self.network = MLPNetwork(num_inputs=obs_space['obs_vec_size'][0] + act_size*(1-is_discrete),
                                      num_outputs=act_size*is_discrete + 1*(1-is_discrete),
                                      hidden_size=hidden_size,
                                      set_final_bias=set_final_bias)

        elif set(obs_space.keys()) == {'obs_map_size', 'obs_vec_size'}:
            assert obs_space['obs_map_size'][1] == obs_space['obs_map_size'][2]  # assumes square image

            self.network = CNN_MLP_hybrid(input_vec_len=obs_space['obs_vec_size'][0] + act_size*(1-is_discrete),
                                          mlp_output_vec_len=act_size*is_discrete + 1*(1-is_discrete),
                                          mlp_hidden_size=hidden_size,
                                          input_maps_size=obs_space['obs_map_size'][1],
                                          num_input_channels=obs_space['obs_map_size'][0],
                                          cnn_output_vec_len=hidden_size,
                                          set_final_bias=set_final_bias)

        else:
            raise NotImplementedError

        self.act_space = act_space
        self.optim = optim.Adam(self.network.parameters(), lr=lr)

    def __call__(self, input):
        return self.network(input)

    @property
    def device(self):
        return next(self.parameters()).device


class DeterministicContinuousPolicy(nn.Module):
    def __init__(self, obs_space, act_space, hidden_size, lr):
        super().__init__()

        self.network = MLPNetwork(num_inputs=obs_space['obs_vec_size'][0],
                                  num_outputs=act_space.shape[0],
                                  hidden_size=hidden_size,
                                  set_final_bias=True)

        self.act_space = act_space
        self.optim = optim.Adam(self.network.parameters(), lr=lr)

    def __call__(self, obs):
        return self.network(obs)

    def act(self, obs, **kwargs):
        return self(obs)

    @property
    def device(self):
        return next(self.parameters()).device


class GaussianPolicy(nn.Module):

    # This a continuous policy

    def __init__(self, obs_space, act_space, hidden_size, lr, action_squashing, set_final_bias=False):
        super().__init__()

        self.network = MLPNetwork(num_inputs=obs_space['obs_vec_size'][0],
                                  num_outputs=act_space.shape[0] * 2,
                                  hidden_size=hidden_size,
                                  set_final_bias=set_final_bias)

        self.optim = optim.Adam(self.network.parameters(), lr=lr)

        self.action_squashing = action_squashing
        if self.action_squashing == 'tanh':
            self.squash_action = torch.tanh
        elif self.action_squashing == 'none':
            self.squash_action = lambda x: x
        else:
            raise NotImplementedError

    def __call__(self, obs):
        return self.network(obs)

    def act(self, obs, sample, return_log_pi):
        return self.act_from_logits(self(obs), sample, return_log_pi)

    def get_log_prob_from_obs_action_pairs(self, action, obs):

        mean, log_std = GaussianPolicy.get_mean_logstd_from_logits(self(obs))
        if self.action_squashing == 'tanh':
            action = torch.distributions.TanhTransform()._inverse(action)

        log_prob = self.log_prob_density(action, mean, log_std)
        return log_prob

    def act_from_logits(self, logits, sample, return_log_pi):
        if return_log_pi:
            return self.get_action_and_log_prob(logits, sample)
        else:
            return self.get_action(logits, sample)

    @staticmethod
    def get_mean_logstd_from_logits(logits):
        mean, log_std = logits.split(logits.shape[1] // 2, dim=-1)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std

    def log_prob_density(self, action, mean, log_std):
        n = len(action) if hasattr(action, '__len__') else 1
        z = (action - mean) / (log_std.exp() + EPS2)

        if self.action_squashing == 'tanh':
            log_prob = (- 0.5 * (2 * log_std + z ** 2 + math.log(2 * math.pi)) \
                        - torch.log(1 - action.tanh() ** 2 + EPS)).sum(dim=-1)
        elif self.action_squashing == 'none':
            log_prob = (- 0.5 * (2 * log_std + z ** 2 + math.log(2 * math.pi))).sum(dim=-1)
        else:
            raise NotImplementedError

        if not n == 1:
            log_prob = log_prob.unsqueeze(-1)
        return log_prob

    def get_action(self, logits, sample):
        mean, log_std = GaussianPolicy.get_mean_logstd_from_logits(logits)
        if not sample:
            action = mean
        else:
            noise = torch.normal(
                torch.zeros_like(mean), torch.ones_like(mean)
            ) * torch.exp(log_std)
            action = mean + noise

        action = self.squash_action(action)
        return action

    def get_action_and_log_prob(self, logits, sample):
        mean, log_std = GaussianPolicy.get_mean_logstd_from_logits(logits)
        noise = torch.normal(
            torch.zeros_like(mean), torch.ones_like(mean)
        ) * torch.exp(log_std)
        action = mean + noise

        log_prob = self.log_prob_density(action, mean, log_std)

        return self.squash_action(action), log_prob

    @property
    def device(self):
        return next(self.parameters()).device


####################### DISCRETE ###########################

class CategoricalPolicy(nn.Module):
    def __init__(self, obs_space, act_space, hidden_size, lr, set_final_bias=False):
        super().__init__()

        if set(obs_space.keys()) == {'obs_vec_size'}:
            self.network = MLPNetwork(num_inputs=obs_space['obs_vec_size'][0],
                                      num_outputs=act_space.n,
                                      hidden_size=hidden_size,
                                      set_final_bias=set_final_bias)

        elif set(obs_space.keys()) == {'obs_map_size', 'obs_vec_size'}:
            assert obs_space['obs_map_size'][1] == obs_space['obs_map_size'][2]  # assumes square image

            self.network = CNN_MLP_hybrid(input_vec_len=obs_space['obs_vec_size'][0],
                                          mlp_output_vec_len=act_space.n,
                                          mlp_hidden_size=hidden_size,
                                          input_maps_size=obs_space['obs_map_size'][1],
                                          num_input_channels=obs_space['obs_map_size'][0],
                                          cnn_output_vec_len=hidden_size,
                                          set_final_bias=set_final_bias)

        else:
            raise NotImplementedError

        self.act_space = act_space
        self.optim = optim.Adam(self.network.parameters(), lr=lr)

    def __call__(self, obs):
        return self.network(obs)

    def act(self, obs, sample, return_log_pi):
        logits = self(obs)
        return self.act_from_logits(logits=logits, sample=sample, return_log_pi=return_log_pi)

    def get_log_prob_from_obs_action_pairs(self, action, obs):
        probas = self(obs=obs)
        return self.log_prob_density(x=action, logits=probas)

    @staticmethod
    def act_from_logits(logits, sample, return_log_pi):
        action = CategoricalPolicy.get_action(logits=logits, sample=sample).detach()
        if return_log_pi:
            log_pi = CategoricalPolicy.log_prob_density(x=action, logits=logits)
            return action, log_pi
        else:
            return action

    @staticmethod
    def get_action(logits, sample):
        if not sample:
            action = logits.argmax(-1)
        else:
            action = gumbel_softmax(logits=logits).detach()
            action = action.argmax(-1)

        return action

    @staticmethod
    def log_prob_density(x, logits):
        log_probas = F.log_softmax(logits, dim=-1)
        log_proba_of_sample = log_probas.gather(dim=1, index=to_torch(x.unsqueeze(1), type=int, device=logits.device))
        return log_proba_of_sample

    @property
    def device(self):
        return next(self.parameters()).device


class DiscreteRandomPolicy(nn.Module):
    def __init__(self, num_out, **kwargs):
        super().__init__()
        self.act_dim = num_out
        self.log_prob = np.log(1. / self.act_dim)

    def act(self, return_log_pi, **kwargs):
        if return_log_pi:
            return np.random.randint(self.act_dim), self.log_prob
        else:
            return np.random.randint(self.act_dim)

    def get_log_prob_from_obs(self, action, obs, alpha=1):
        probas = torch.full((len(obs), 1), fill_value=self.log_prob)
        return probas


###################### MISC ############################################


class V(nn.Module):
    def __init__(self, obs_space, hidden_size, lr, set_final_bias=False):
        super().__init__()

        if set(obs_space.keys()) == {'obs_vec_size'}:
            self.network = MLPNetwork(num_inputs=obs_space['obs_vec_size'][0],
                                      num_outputs=1,
                                      hidden_size=hidden_size,
                                      set_final_bias=set_final_bias)

        elif set(obs_space.keys()) == {'obs_map_size', 'obs_vec_size'}:
            assert obs_space['obs_map_size'][1] == obs_space['obs_map_size'][2]

            self.network = CNN_MLP_hybrid(input_vec_len=obs_space['obs_vec_size'][0],
                                          mlp_output_vec_len=1,
                                          mlp_hidden_size=hidden_size,
                                          input_maps_size=obs_space['obs_map_size'][1],
                                          num_input_channels=obs_space['obs_map_size'][0],
                                          cnn_output_vec_len=hidden_size,
                                          set_final_bias=set_final_bias)

        else:
            raise NotImplementedError

        self.optim = optim.Adam(self.network.parameters(), lr=lr)

    def __call__(self, input):
        return self.network(input)

    @property
    def device(self):
        return next(self.parameters()).device


class ParameterAsModel(nn.Module):
    def __init__(self, data, requires_grad):
        super().__init__()
        self.value = nn.Parameter(data, requires_grad=requires_grad)

    @property
    def device(self):
        return next(self.parameters()).device
