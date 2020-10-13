from direct_rl.ppo import PPOLearner
from direct_rl.sac import SACLearner
from direct_rl.sacmh import SACMHLearner
from direct_rl.random_agent import RandomLearner
from irl.gail import GAILLearner
from irl.ail import AILLearner
from irl.asaf import ASAFLearner
from irl.bc import BCLearner
from alg_task_lists import RL_ALGS, IRL_ALGS, RL_OFF_POLICY_ALGS, ALGS

import torch


def get_corresponding_train_script(alg_names):
    assert all([alg_name in ALGS for alg_name in alg_names]), "Some alg_name are not defined in alg_task_lists.py"

    if all([alg_name in RL_ALGS for alg_name in alg_names]):
        import direct_rl.train as train_script

    elif all([alg_name in IRL_ALGS for alg_name in alg_names]):
        import irl.train as train_script

    else:
        raise ValueError("'alg_names' contains a mix of RL and IRL algorithms."
                         "Since RL_ALGS and IRL_ALGS don't use the same train.py and hyperparameters, "
                         "searches must be defined on only one of these categories of algorithm at a time.")

    return train_script


def init_from_save(filepath, device):
    save_dict = torch.load(filepath, map_location=device)
    alg_name = save_dict['init_dict']['alg_name']
    if alg_name in ["sacmh", "sqil"]:
        algo = SACMHLearner.init_from_save(filepath, device)
    elif alg_name in ["sac", "sqil-c"]:
        algo = SACLearner.init_from_save(filepath, device)
    elif alg_name == "ppo":
        algo = PPOLearner.init_from_save(filepath, device)
    elif alg_name == "random":
        algo = RandomLearner.init_from_save(filepath, device)
    elif alg_name == "gail":
        algo = GAILLearner.init_from_save(filepath, device)
    elif alg_name in ["asqf", "airl"]:
        algo = AILLearner.init_from_save(filepath, device)
    elif alg_name in ["asaf-full", "asaf-1", "asaf-w"]:
        algo = ASAFLearner.init_from_save(filepath, device)
    elif alg_name == "bc":
        algo = BCLearner.init_from_save(filepath, device)
    else:
        raise NotImplementedError(f"{alg_name} is not a recognized alg_name")
    return algo


def named_init_from_config(env_dims, config, alg_name):
    if alg_name == "ppo":
        algorithm = PPOLearner(obs_space=env_dims['obs_space'],
                               act_space=env_dims['act_space'],
                               hidden_size=config.hidden_size,
                               lr=config.lr,
                               critic_lr_coef=config.critic_lr_coef,
                               batch_size=config.batch_size,
                               epochs_per_update=config.epochs_per_update,
                               update_clip_param=config.update_clip_param,
                               grad_norm_clip=config.grad_norm_clip,
                               gamma=config.gamma,
                               lamda=config.lamda,
                               critic_loss_coeff=config.critic_loss_coeff)

    elif alg_name in ["sac", "sqil-c"]:
        algorithm = SACLearner(obs_space=env_dims['obs_space'],
                               act_space=env_dims['act_space'],
                               hidden_size=config.hidden_size,
                               lr=config.lr,
                               init_alpha=config.init_alpha,
                               gamma=config.gamma,
                               grad_norm_clip=config.grad_norm_clip,
                               tau=config.tau,
                               alg_name=alg_name)

    elif alg_name in ["sacmh", "sqil"]:
        algorithm = SACMHLearner(obs_space=env_dims['obs_space'],
                                 act_space=env_dims['act_space'],
                                 hidden_size=config.hidden_size,
                                 lr=config.lr,
                                 init_alpha=config.init_alpha,
                                 gamma=config.gamma,
                                 grad_norm_clip=config.grad_norm_clip,
                                 tau=config.tau,
                                 alg_name=alg_name)

    elif alg_name == "random":
        algorithm = RandomLearner(action_dim=env_dims['act_space'])

    elif alg_name == "gail":
        discriminator_args = {'hidden_size': config.hidden_size}

        algorithm = GAILLearner(obs_space=env_dims['obs_space'],
                                act_space=env_dims['act_space'],
                                discriminator_args=discriminator_args,
                                discriminator_lr=config.d_lr,
                                reward_definition=config.d_reward_definition,
                                grad_norm_clip=config.d_grad_norm_clip,
                                gradient_penalty_coef=config.gradient_penalty_coef,
                                alg_name=alg_name)

    elif alg_name in ["asqf", "airl"]:
        discriminator_args = {k: config.__dict__[k] for k in
                              ('hidden_size', 'use_advantage_formulation', 'use_multi_head')}
        algorithm = AILLearner(obs_space=env_dims['obs_space'],
                               act_space=env_dims['act_space'],
                               discriminator_args=discriminator_args,
                               discriminator_lr=config.d_lr,
                               grad_norm_clip=config.d_grad_norm_clip,
                               alg_name=alg_name)

    elif alg_name in ["asaf-full", "asaf-1", "asaf-w"]:
        discriminator_args = {'hidden_size': config.hidden_size, 'lr': config.d_lr}

        algorithm = ASAFLearner(obs_space=env_dims['obs_space'],
                                act_space=env_dims['act_space'],
                                grad_value_clip=config.d_grad_value_clip,
                                grad_norm_clip=config.d_grad_norm_clip,
                                discriminator_args=discriminator_args,
                                break_traj_to_windows=config.break_traj_to_windows,
                                window_size=config.window_size,
                                window_stride=config.window_stride,
                                window_over_episode=config.window_over_episode,
                                alg_name=alg_name)

    elif alg_name == "bc":
        algorithm = BCLearner(obs_space=env_dims['obs_space'],
                              act_space=env_dims['act_space'],
                              lr=config.d_lr,
                              hidden_size=config.hidden_size)

    elif alg_name == "":
        return

    else:
        raise NotImplementedError(f"{alg_name} is not a recognized alg_name")

    return algorithm


def init_from_config(env_dims, config):
    # Initializes agents
    alg_name = config.alg_name
    return named_init_from_config(env_dims, config, alg_name)


def should_update_rl(episode, done, n_transitions, total_transitions, config, name, did_irl_update_first=True):
    if not did_irl_update_first:
        return False

    if name in RL_OFF_POLICY_ALGS and n_transitions < config.warmup:
        return False

    elif config.episodes_between_updates is not None:
        return done and episode % config.episodes_between_updates == 0 and n_transitions >= config.batch_size

    elif config.transitions_between_updates is not None:
        return total_transitions % config.transitions_between_updates == 0

    else:
        raise ValueError('Choose between episode-wise or transition-wise training')


def should_update_on_policy_irl(episode, done, n_transitions, total_transitions, config):
    assert (config.d_episodes_between_updates is not None) or (config.d_transitions_between_updates is not None), \
        "Choose between episode-wise or transition-wise training"

    if config.d_episodes_between_updates is not None:
        # Note that, except for asaf-1, the last condition is misleading as it compares transitions
        # with trajectories or windows
        return done and episode % config.d_episodes_between_updates == 0 and n_transitions >= config.d_batch_size

    elif config.d_transitions_between_updates is not None:
        return total_transitions % config.d_transitions_between_updates == 0

    else:
        raise ValueError('Choose between episode-wise or transition-wise training')


def overide_args(alg_name, args):
    if alg_name == 'ppo':
        args.__dict__.update(DEFAULT_PPO_ARGS)

    elif alg_name in ["sac", "sqil-c"]:
        args.__dict__.update(DEFAULT_SAC_ARGS)

    elif alg_name in ['sacmh', 'sqil']:
        args.__dict__.update(DEFAULT_SACMH_ARGS)

    elif alg_name == "random":
        args.__dict__.update(DEFAULT_RANDOM)

    elif alg_name == '':
        pass

    elif alg_name in ['asqf']:
        args.__dict__.update(DEFAULT_ASQF_ARGS)

    elif alg_name in ["asaf-full"]:
        args.__dict__.update(DEFAULT_ASAF_ARGS)

    elif alg_name in ["asaf-1"]:
        args.__dict__.update(DEFAULT_ASAF1_ARGS)

    elif alg_name in ["asaf-w"]:
        args.__dict__.update(DEFAULT_ASAFW_ARGS)

    elif alg_name == 'airl':
        args.__dict__.update(DEFAULT_AIRL_ARGS_ON_POLICY)

    elif alg_name == 'bc':
        args.__dict__.update(DEFAULT_BC_ARGS)

    elif alg_name == 'gail':
        args.__dict__.update(DEFAULT_GAIL_ARGS)

    else:
        raise NotImplementedError(f"{alg_name} is not a recognized alg_name")

    if alg_name in ['sac', 'ppo']:
        if args.task_name == 'hopper-c':
            args.max_episodes = None
            args.episodes_between_saves = None
            args.max_transitions = int(1e6)
            args.transitions_between_saves = int(1e3)
        elif args.task_name == 'walker-c':
            args.max_episodes = None
            args.episodes_between_saves = None
            args.max_transitions = int(3e6)
            args.transitions_between_saves = int(1e3)
        elif args.task_name == 'halfcheetak-c':
            args.max_episodes = None
            args.episodes_between_saves = None
            args.max_transitions = int(3e6)
            args.transitions_between_saves = int(1e3)
        elif args.task_name == 'ant-c':
            args.max_episodes = None
            args.episodes_between_saves = None
            args.max_transitions = int(3e6)
            args.transitions_between_saves = int(1e3)
        elif args.task_name == 'humanoid-c':
            args.max_episodes = None
            args.episodes_between_saves = None
            args.max_transitions = int(1e7)
            args.transitions_between_saves = int(1e3)

    return args


DEFAULT_PPO_ARGS = {}

DEFAULT_SAC_ARGS = {}

DEFAULT_SACMH_ARGS = {}

DEFAULT_BC_ARGS = {}

DEFAULT_GAIL_ARGS = {}

DEFAULT_ASQF_ARGS = {}

DEFAULT_ASAF_ARGS = {}

DEFAULT_ASAFW_ARGS = {}

DEFAULT_ASAF1_ARGS = {}

DEFAULT_AIRL_ARGS_ON_POLICY = {}

DEFAULT_RANDOM = {}
DEFAULT_RANDOM.update(DEFAULT_SACMH_ARGS)
