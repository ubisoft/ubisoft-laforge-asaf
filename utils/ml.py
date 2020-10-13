import numpy as np
import random
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from gym import spaces


def set_seeds(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_env_dims(env):
    try:
        obs_space = {obs_name: obs_space_i.shape for obs_name, obs_space_i in env.observation_space.items()}
    except AttributeError:
        obs_space = env.observation_space
    act_space = env.action_space
    return {'obs_space': obs_space, 'act_space': act_space}


def get_act_properties_from_act_space(act_space):
    if isinstance(act_space, spaces.Box):
        act_size = int(np.prod(act_space.shape))
        is_discrete = int(False)
    elif isinstance(act_space, spaces.Discrete):
        act_size = act_space.n
        is_discrete = int(True)
    else:
        raise NotImplementedError

    return act_size, is_discrete


def discrete_from_one_hot(one_hot):
    if isinstance(one_hot, np.ndarray):
        return np.argmax(one_hot, -1)
    else:
        return torch.argmax(one_hot, -1, keepdim=True)


def data_to_device(data, device):
    if device is not None:
        data.to(device)


def to_torch(data, type=None, device=None, requires_grad=False):
    # casts data to corresponding torch type

    if device is None:  # we take the default device specified by pytorch
        if type is None:
            if 'float' in data.dtype.name:
                returned_data = torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)
            elif 'int' in data.dtype.name:  # pytorch ints must be long ints as compared to numpy
                returned_data = torch.tensor(data, dtype=torch.long, requires_grad=requires_grad)
            else:
                raise NotImplementedError
        else:
            returned_data = torch.tensor(data, dtype=type, requires_grad=requires_grad)
    else:
        if type is None:
            if 'float' in data.dtype.name:
                returned_data = torch.tensor(data, dtype=torch.float32, device=device, requires_grad=requires_grad)
            elif 'int' in data.dtype.name:
                returned_data = torch.tensor(data, dtype=torch.long, device=device, requires_grad=requires_grad)
            else:
                raise NotImplementedError
        else:
            returned_data = torch.tensor(data, dtype=type, device=device, requires_grad=requires_grad)

    return returned_data


def to_device(model, device):
    if not model.device == device:
        data_to_device(data=model, device=device)


def to_numpy(data):
    return data.data.cpu().numpy()


def save_checkpoint(state, filename):
    torch.save(state, filename)


def onehot_from_index(index, onehot_size):
    if hasattr(index, '__len__'):
        one_hots = np.zeros((len(index), onehot_size))
        one_hots[np.arange(len(index)), index] = 1
    else:
        one_hots = np.zeros(onehot_size)
        one_hots[index] = 1
    return one_hots


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(same_as, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.zeros_like(same_as, requires_grad=False)
    U.data.uniform_()
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(same_as=logits)
    return F.softmax(y / temperature, dim=-1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs (but divided by alpha)
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def save(models, directory, suffix):
    if not directory is None:
        for model in models:
            model.save(f"{directory / model.name}_{suffix}")


def wandb_watch(wandb, learners):
    to_watch = []
    for learner in learners:
        to_watch += learner.wandb_watchable()

    if len(to_watch) > 0:
        wandb.watch(to_watch, log="all")


def remove(models, directory, suffix):
    for model in models:
        os.remove(f"{directory / model.name}_{suffix}")


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    assert 0. < tau and tau < 1.
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def mask(done):
    return 0 if done else 1

