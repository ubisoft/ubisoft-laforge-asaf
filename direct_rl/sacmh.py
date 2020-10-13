from base_algo import BaseAlgo
from direct_rl.models import Q, CategoricalPolicy, ParameterAsModel
from utils.ml import soft_update, hard_update, to_torch, to_numpy, to_device, log_sum_exp
from utils.obs_dict import obs_to_torch

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

MSELoss = torch.nn.MSELoss()

class SACMHLearner(BaseAlgo):
    """
    Multi-Head SAC (soft actor-critic for discrete control)
    """

    def __init__(self, obs_space, act_space, hidden_size, lr, init_alpha, grad_norm_clip, gamma, tau, alg_name):
        super().__init__()

        self.q1 = Q(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)
        self.q2 = Q(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)

        with torch.no_grad():
            self.tq1 = Q(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)
            self.tq2 = Q(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr)

        hard_update(target=self.tq1, source=self.q1)
        hard_update(target=self.tq2, source=self.q2)

        self.log_alpha = ParameterAsModel(torch.tensor(np.log(init_alpha)), requires_grad=True)
        self.target_entropy = 0.1
        self.alpha_optimizer = optim.Adam(self.log_alpha.parameters(), lr=lr)

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
                          'alg_name': self.name}

        self.metrics_to_record = {'total_transitions', 'eval_step', 'episode', 'episode_len', 'epoch', 'q1_loss',
                                  'q2_loss', 'alpha_loss', 'q_s', 'return', 'eval_return', 'cumul_eval_return',
                                  'avg_eval_return', 'pi_entropy', 'alpha'}

    def train_model(self, experiences):
        observations = obs_to_torch(experiences[0], device=self.device)
        actions = experiences[1]
        next_states = obs_to_torch(experiences[2], device=self.device)
        rewards = to_torch(experiences[3], device=self.device)
        masks = to_torch(experiences[4], device=self.device)

        q1_s = self.q1(observations)
        q2_s = self.q2(observations)

        q_s = torch.min(q1_s, q2_s)

        alpha = self.log_alpha.value.exp()

        alpha_loss = (-(F.softmax(q_s / alpha, dim=1) * q_s).sum(1) + alpha * (
                - self.target_entropy + log_sum_exp(q_s / alpha, dim=1, keepdim=False)))
        alpha_loss = alpha_loss.mean(0)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.log_alpha.parameters(), self.grad_norm_clip)
        self.alpha_optimizer.step()

        # q values of current state action pairs

        q1_s_a = q1_s.gather(dim=1, index=to_torch(actions, type=int, device=self.device)).squeeze(1)
        q2_s_a = q2_s.gather(dim=1, index=to_torch(actions, type=int, device=self.device)).squeeze(1)

        # # target q values

        q1_sp = self.tq1(next_states)
        q2_sp = self.tq2(next_states)

        target_q = torch.min(q1_sp, q2_sp)

        pi_entropy = (- F.softmax(target_q / alpha, dim=1) * F.log_softmax(target_q / alpha, dim=1)).sum(1)
        target = rewards + masks * self.gamma * (
                (F.softmax(target_q / alpha, dim=1) * target_q).sum(1) + alpha * pi_entropy)

        # losses

        q1_loss = ((q1_s_a - target.detach()) ** 2).mean(0)
        q2_loss = ((q2_s_a - target.detach()) ** 2).mean(0)

        # backprop

        self.q1.optim.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_norm_clip)
        self.q1.optim.step()

        self.q2.optim.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_norm_clip)
        self.q2.optim.step()

        return_dict = {}
        return_dict.update({'q1_loss': to_numpy(q1_loss),
                            'q2_loss': to_numpy(q2_loss),
                            'alpha_loss': to_numpy(alpha_loss),
                            'pi_entropy': to_numpy(pi_entropy.mean()),
                            'q_s': to_numpy(q_s.mean()),
                            'alpha': to_numpy(alpha)})

        # update targets networks

        self.update_targets()

        return return_dict

    def update_targets(self):
        soft_update(target=self.tq1, source=self.q1, tau=self.tau)
        soft_update(target=self.tq2, source=self.q2, tau=self.tau)

    def act(self, obs, sample):
        q_s = self(obs_to_torch(obs, unsqueeze_dim=0, device=self.device))
        return int(CategoricalPolicy.act_from_logits(
            logits=q_s / self.log_alpha.value.exp(), sample=sample, return_log_pi=False).data.cpu().numpy())

    def __call__(self, obs):
        return torch.min(self.q1(obs), self.q2(obs))

    def log_prob_density(self, x, logits):
        return CategoricalPolicy.log_prob_density(x, logits / self.log_alpha.value.exp())

    def get_log_prob_from_obs_action_pairs(self, action, obs):
        logits = self(obs=obs)
        return self.log_prob_density(x=action, logits=logits / self.log_alpha.value.exp())

    def get_policy(self):
        return self

    def prep_rollout(self, rollout_device):
        self.q1.eval()
        self.q2.eval()
        to_device(self.q1, rollout_device)
        to_device(self.q2, rollout_device)
        to_device(self.log_alpha, rollout_device)

    def prep_training(self, train_device):
        self.q1.train()
        self.q2.train()
        to_device(self.q1, train_device)
        to_device(self.q2, train_device)
        to_device(self.tq1, train_device)
        to_device(self.tq2, train_device)
        to_device(self.log_alpha, train_device)

    @property
    def device(self):
        assert self.q1.device == self.q2.device
        assert self.q1.device == self.log_alpha.device
        return self.q1.device

    def get_params(self):
        return {'q1': self.q1.state_dict(),
                'q2': self.q2.state_dict(),
                'tq1': self.tq1.state_dict(),
                'tq2': self.tq2.state_dict()}

    def load_params(self, params):
        self.q1.load_state_dict(params['q1'])
        self.q2.load_state_dict(params['q2'])
        self.tq1.load_state_dict(params['tq1'])
        self.tq2.load_state_dict(params['tq2'])

    def save_training_graphs(self, train_recorder, save_dir):
        from alfred.utils.plots import create_fig, plot_curves
        import matplotlib.pyplot as plt

        # Losses

        fig, axes = create_fig((4, 1))
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
                    ys=[train_recorder.tape['alpha_loss']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel="Transitions",
                    ylabel="alpha_loss")
        plot_curves(axes[3],
                    ys=[train_recorder.tape['q_s']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel="Transitions",
                    ylabel="q_s")

        fig.savefig(str(save_dir / 'losses.png'))
        plt.close(fig)

        # True Returns
        fig, axes = create_fig((4, 1))
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
                    xlabel="Transitions", ylabel="pi_entropy")
        plot_curves(axes[3],
                    ys=[train_recorder.tape['alpha']],
                    xs=[train_recorder.tape['total_transitions']],
                    xlabel="Transitions", ylabel="alpha")
        fig.savefig(str(save_dir / 'figures.png'))
        plt.close(fig)

    def wandb_watchable(self):
        return [self.q1.network, self.q2.network]
