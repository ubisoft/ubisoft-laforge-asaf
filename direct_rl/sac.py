from base_algo import BaseAlgo
from direct_rl.models import Q, GaussianPolicy
from utils.ml import soft_update, hard_update, to_torch, to_device
from utils.obs_dict import obs_to_torch

import torch
import math
import numpy as np

MSELoss = torch.nn.MSELoss()


class SACLearner(BaseAlgo):
    """
    Soft Actor-Critic for Continuous Control
    """

    def __init__(self, obs_space, act_space, hidden_size, lr, init_alpha, grad_norm_clip, gamma, tau, alg_name, learn_alpha=True):
        super().__init__()

        self.pi = GaussianPolicy(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr,
                                 action_squashing='tanh', set_final_bias=True)

        self.q1 = Q(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)
        self.q2 = Q(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)

        with torch.no_grad():
            self.tq1 = Q(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)
            self.tq2 = Q(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)

        hard_update(target=self.tq1, source=self.q1)
        hard_update(target=self.tq2, source=self.q2)

        self.learn_alpha = learn_alpha
        if self.learn_alpha:
            self._log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True)
            self._alpha = torch.exp(self._log_alpha.detach())
            self.alpha_optimizer = torch.optim.Adam((self._log_alpha,), lr=lr)
            self.target_entropy = - np.prod(act_space.shape)
        else:
            self._alpha = init_alpha

        self.grad_norm_clip = grad_norm_clip
        self.gamma = gamma
        self.tau = tau

        self.name = alg_name
        self.init_dict = {'obs_space': obs_space,
                          'act_space': act_space,
                          'hidden_size': hidden_size,
                          'lr': lr,
                          'grad_norm_clip': grad_norm_clip,
                          'gamma': gamma,
                          'tau': tau,
                          'init_alpha': init_alpha,
                          'learn_alpha': learn_alpha,
                          'alg_name': self.name}

        self.metrics_to_record = {'total_transitions', 'eval_step', 'episode', 'episode_len', 'epoch', 'loss', 'return',
                                  'eval_return', 'cumul_eval_return', 'avg_eval_return',
                                  'alpha_loss', 'pi_loss', 'q1_loss', 'q2_loss', 'pi_entropy'}

    def train_model(self, experiences):
        obs, act, new_obs, rew, mask = experiences
        obs, new_obs = [obs_to_torch(o, device=self.device) for o in (obs, new_obs)]
        act, rew, mask = [to_torch(e, device=self.device) for e in (act, rew, mask)]

        # Define critic loss.
        cat = torch.cat((obs['obs_vec'], act), dim=-1)

        q1 = self.q1(cat).squeeze(1)  # N
        q2 = self.q2(cat).squeeze(1)  # N

        with torch.no_grad():
            new_act, new_log_prob = self.pi.act(new_obs, sample=True, return_log_pi=True)
            new_cat = torch.cat((new_obs['obs_vec'], new_act), dim=-1).detach()
            tq1 = self.tq1(new_cat).squeeze(1)  # N
            tq2 = self.tq2(new_cat).squeeze(1)  # N

        tq = torch.min(tq1, tq2)  # N
        # print(rew.shape, mask.shape, tq.shape, self._alpha, new_log_prob.shape)
        target = rew + self.gamma * mask * (tq - self._alpha * new_log_prob)

        q1_loss = (0.5 * (target - q1) ** 2).mean()
        q2_loss = (0.5 * (target - q2) ** 2).mean()

        # Define actor loss.
        act2, log_prob2 = self.pi.act(obs, sample=True, return_log_pi=True)
        cat2 = torch.cat((obs['obs_vec'], act2), dim=-1)

        q1 = self.q1(cat2).squeeze(1)  # N
        q2 = self.q2(cat2).squeeze(1)  # N
        q = torch.min(q1, q2)  # N

        pi_loss = (self._alpha * log_prob2 - q).mean()

        # Update non-target networks.
        if self.learn_alpha:
            # Define alpha loss
            alpha_loss = - self._log_alpha * (log_prob2.detach() + self.target_entropy).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self._alpha = torch.exp(self._log_alpha).detach()

        self.pi.optim.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_norm_clip)
        self.pi.optim.step()

        self.q1.optim.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_norm_clip)
        self.q1.optim.step()

        self.q2.optim.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_norm_clip)
        self.q2.optim.step()

        # Update target networks.
        self.update_targets()

        return {
            'alpha_loss': alpha_loss.detach().cpu().numpy(),
            'pi_loss': pi_loss.detach().cpu().numpy(),
            'q1_loss': q1_loss.detach().cpu().numpy(),
            'q2_loss': q2_loss.detach().cpu().numpy(),
            'pi_entropy': - log_prob2.mean().detach().cpu().numpy(),
        }

    def update_targets(self):
        soft_update(target=self.tq1, source=self.q1, tau=self.tau)
        soft_update(target=self.tq2, source=self.q2, tau=self.tau)

    @torch.no_grad()
    def act(self, obs, sample):
        obs = obs_to_torch(obs, device=self.device, unsqueeze_dim=0)
        act = self.pi.act(obs, sample, return_log_pi=False).detach()
        return act[0].cpu().numpy()

    def __call__(self, obs):
        return torch.min(self.q1(obs), self.q2(obs))

    def get_policy(self):
        return self

    def prep_rollout(self, rollout_device):
        self.pi.eval()
        self.q1.eval()
        self.q2.eval()
        to_device(self.pi, rollout_device)
        to_device(self.q1, rollout_device)
        to_device(self.q2, rollout_device)

    def prep_training(self, train_device):
        self.pi.train()
        self.q1.train()
        self.q2.train()
        to_device(self.pi, train_device)
        to_device(self.q1, train_device)
        to_device(self.q2, train_device)
        to_device(self.tq1, train_device)
        to_device(self.tq2, train_device)

    @property
    def device(self):
        assert self.q1.device == self.q2.device
        return self.q1.device

    def get_params(self):
        return {
            'pi': self.pi.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'tq1': self.tq1.state_dict(),
            'tq2': self.tq2.state_dict()
        }

    def load_params(self, params):
        self.pi.load_state_dict(params['pi'])
        self.q1.load_state_dict(params['q1'])
        self.q2.load_state_dict(params['q2'])
        self.tq1.load_state_dict(params['tq1'])
        self.tq2.load_state_dict(params['tq2'])

    def save_training_graphs(self, train_recorder, save_dir):
        from alfred.utils.plots import create_fig, plot_curves
        import matplotlib.pyplot as plt

        # Losses

        fig, axes = create_fig((3, 1))
        plot_curves(axes[0],
                    ys=[train_recorder.tape['q1_loss']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel='Transitions',
                    ylabel='q1_loss')
        plot_curves(axes[1],
                    ys=[train_recorder.tape['q2_loss']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel='Transitions',
                    ylabel='q2_loss')
        plot_curves(axes[2],
                    ys=[train_recorder.tape['pi_loss']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel='Transitions',
                    ylabel='pi_loss')

        fig.savefig(str(save_dir / 'losses.png'))
        plt.close(fig)

        # True Returns
        fig, axes = create_fig((3, 1))
        fig.suptitle('Returns')
        plot_curves(axes[0],
                    ys=[train_recorder.tape['return']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel='Transitions', ylabel="Mean Return")
        plot_curves(axes[1],
                    ys=[train_recorder.tape['eval_return']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel='Transitions', ylabel="Mean Eval Return")
        plot_curves(axes[2],
                    ys=[train_recorder.tape['pi_entropy']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel='Transitions', ylabel="pi_entropy")
        fig.savefig(str(save_dir / 'figures.png'))
        plt.close(fig)

    def wandb_watchable(self):
        return [self.pi.network, self.q1.network, self.q2.network]
