from alfred.utils.recorder import TrainingIterator

from base_algo import BaseAlgo
from direct_rl.models import CategoricalPolicy, V, GaussianPolicy
from utils.ml import to_torch, to_numpy, to_device
from utils.obs_dict import ObsDict, obs_to_torch
from gym.spaces import Discrete, Box

import torch
from torch import nn
import numpy as np


class PPOLearner(BaseAlgo):
    def __init__(self, obs_space, act_space, hidden_size, lr, critic_lr_coef, batch_size, epochs_per_update,
                 update_clip_param, gamma, lamda, grad_norm_clip=1e8, alg_name='ppo', critic_loss_coeff=0.5):
        assert alg_name == 'ppo'
        super().__init__()

        if isinstance(act_space, Discrete):
            self.actor = CategoricalPolicy(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)
            self.discrete = True
        elif isinstance(act_space, Box):
            self.actor = GaussianPolicy(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr,
                                        action_squashing='none', set_final_bias=True)
            self.discrete = False
        else:
            raise NotImplementedError

        self.critic = V(obs_space=obs_space, hidden_size=hidden_size, lr=lr * critic_lr_coef, set_final_bias=True)
        self.name = alg_name
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        self.update_clip_param = update_clip_param
        self.gamma = gamma
        self.lamda = lamda
        self.grad_norm_clip = grad_norm_clip
        self.critic_loss_coeff = critic_loss_coeff
        self.init_dict = {'obs_space': obs_space,
                          'act_space': act_space,
                          'hidden_size': hidden_size,
                          'lr': lr,
                          'critic_lr_coef': critic_lr_coef,
                          'batch_size': batch_size,
                          'epochs_per_update': epochs_per_update,
                          'update_clip_param': update_clip_param,
                          'gamma': gamma,
                          'lamda': lamda,
                          'grad_norm_clip': grad_norm_clip,
                          'critic_loss_coeff': critic_loss_coeff,
                          'alg_name': alg_name}

        self.metrics_to_record = {'total_transitions', 'eval_transition', 'episode', 'episode_len', 'epoch', 'loss',
                                  'return', 'eval_return', 'cumul_eval_return', 'avg_eval_return'}

        for k in ['mean', 'max', 'min', 'std', 'median']:
            self.metrics_to_record.add(f'ppo_rewards_{k}')

    def train_model(self, experiences):
        observations = obs_to_torch(obs=experiences[0], device=self.device)
        actions = to_torch(data=experiences[1], device=self.device)
        next_observations = obs_to_torch(obs=experiences[2], device=self.device)
        rewards = experiences[3]
        masks = experiences[4]

        old_values = self.critic(observations)
        old_next_values = self.critic(next_observations)
        returns, advants = self.get_gae(rewards=rewards,
                                        masks=masks,
                                        values=old_values.detach(),
                                        next_values=old_next_values.detach())

        old_policy = self.actor.get_log_prob_from_obs_action_pairs(action=actions, obs=observations)

        criterion = torch.nn.MSELoss()
        n = len(observations)

        for it_update in TrainingIterator(self.epochs_per_update):
            shuffled_idxs = torch.randperm(n, device=self.device)

            for i, it_batch in enumerate(TrainingIterator(n // self.batch_size)):
                batch_idxs = shuffled_idxs[self.batch_size * i: self.batch_size * (i + 1)]

                inputs = ObsDict({obs_name: obs_value[batch_idxs] for obs_name, obs_value in observations.items()})
                actions_samples = actions[batch_idxs]
                returns_samples = returns.unsqueeze(1)[batch_idxs]
                advants_samples = advants.unsqueeze(1)[batch_idxs]
                oldvalue_samples = old_values[batch_idxs].detach()

                values = self.critic(inputs)
                clipped_values = oldvalue_samples + \
                                 torch.clamp(values - oldvalue_samples,
                                             -self.update_clip_param,
                                             self.update_clip_param)
                critic_loss1 = criterion(clipped_values, returns_samples)
                critic_loss2 = criterion(values, returns_samples)
                critic_loss = torch.max(critic_loss1, critic_loss2)

                loss, ratio = self.surrogate_loss(advants_samples, inputs, old_policy.detach(), actions_samples,
                                                  batch_idxs)
                clipped_ratio = torch.clamp(ratio,
                                            1.0 - self.update_clip_param,
                                            1.0 + self.update_clip_param)
                clipped_loss = clipped_ratio * advants_samples
                actor_loss = -torch.min(loss, clipped_loss).mean(0)

                loss = actor_loss + self.critic_loss_coeff * critic_loss

                self.critic.optim.zero_grad()
                self.actor.optim.zero_grad()

                loss.backward()
                nn.utils.clip_grad_norm_([p for p in self.actor.parameters()] + [p for p in self.critic.parameters()],
                                         self.grad_norm_clip)

                self.critic.optim.step()
                self.actor.optim.step()
                it_update.record('loss', to_numpy(loss))

        vals = it_update.pop('loss')
        return_dict = {'loss': np.mean(vals)}

        return_dict.update({
            'ppo_rewards_mean': np.mean(rewards),
            'ppo_rewards_max': np.max(rewards),
            'ppo_rewards_min': np.min(rewards),
            'ppo_rewards_std': np.std(rewards),
            'ppo_rewards_median': np.median(rewards),
        })
        return return_dict

    def get_policy(self):
        return self.actor

    def get_gae(self, rewards, masks, values, next_values):
        rewards = to_torch(rewards, device=self.device)
        masks = to_torch(masks, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)
        advants = torch.zeros_like(rewards, device=self.device)

        running_returns = next_values[-1] * masks[-1]
        previous_value = next_values[-1] * masks[-1]
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + (self.gamma * running_returns * masks[t])
            returns[t] = running_returns

            running_delta = rewards[t] + (self.gamma * previous_value * masks[t]) - values[t]
            previous_value = values[t]

            running_advants = running_delta + (self.gamma * self.lamda * running_advants * masks[t])
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def surrogate_loss(self, advants, states, old_policy, actions, batch_index):
        new_policy = self.actor.get_log_prob_from_obs_action_pairs(action=actions,
                                                                   obs=states)
        old_policy = old_policy[batch_index]

        ratio = torch.exp(new_policy - old_policy)
        surrogate_loss = ratio * advants

        return surrogate_loss, ratio

    def act(self, obs, sample):
        obs = obs_to_torch(obs, unsqueeze_dim=0, device=self.device)
        action = self.actor.act(obs, sample=sample, return_log_pi=False)[0].data.cpu().numpy()
        if self.discrete:
            return int(action)
        else:
            return action

    def prep_rollout(self, rollout_device):
        self.actor.eval()
        to_device(model=self.actor, device=rollout_device)

    def prep_training(self, train_device):
        self.actor.train()
        self.critic.train()
        to_device(self.actor, train_device)
        to_device(self.critic, train_device)

    @property
    def device(self):
        return self.actor.device

    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'actor_optim': self.actor.optim.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_optim': self.critic.optim.state_dict()}

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.actor.optim.load_state_dict(params['actor_optim'])
        self.critic.load_state_dict(params['critic'])
        self.critic.optim.load_state_dict(params['critic_optim'])

    def save_training_graphs(self, train_recorder, save_dir):
        from alfred.utils.plots import create_fig, plot_curves
        import matplotlib.pyplot as plt

        # Loss and return

        fig, axes = create_fig((3, 1))
        plot_curves(axes[0],
                    ys=[train_recorder.tape['loss']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel='Transitions',
                    ylabel="loss")
        plot_curves(axes[1],
                    ys=[train_recorder.tape['return']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel="Transitions",
                    ylabel="return")
        plot_curves(axes[2],
                    ys=[train_recorder.tape['eval_return']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel="Transitions",
                    ylabel="Eval return")

        fig.savefig(str(save_dir / 'figures.png'))
        plt.close(fig)

    def wandb_watchable(self):
        return [self.actor.network, self.critic.network]
