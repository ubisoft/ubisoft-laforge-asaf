from alfred.utils.recorder import TrainingIterator

from base_algo import BaseAlgo
from direct_rl.models import CategoricalPolicy, DeterministicContinuousPolicy
from utils.ml import to_torch, to_numpy, to_device
from utils.data_structures import batch_iter
from utils.obs_dict import obs_to_torch

import torch
import math
from gym.spaces import Box, Discrete

CELoss = torch.nn.CrossEntropyLoss()
MSELoss = torch.nn.MSELoss()


class BCLearner(BaseAlgo):
    def __init__(self, obs_space, act_space, lr, hidden_size, alg_name="bc"):
        super().__init__()
        assert alg_name == "bc"
        self.obs_space = obs_space
        self.act_space = act_space

        if isinstance(act_space, Discrete):
            self.discrete = True
            self.classifier = CategoricalPolicy(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size,
                                                lr=lr)
            self.loss_function = CELoss
        elif isinstance(act_space, Box):
            self.discrete = False
            self.classifier = DeterministicContinuousPolicy(obs_space=obs_space, act_space=act_space,
                                                            hidden_size=hidden_size, lr=lr)
            self.loss_function = MSELoss

        else:
            raise NotImplementedError

        self.name = alg_name
        self.expert_paths = None
        self.init_dict = {'obs_space': obs_space,
                          'act_space': act_space,
                          'lr': lr,
                          'hidden_size': hidden_size,
                          'alg_name': alg_name}

        self.metrics_to_record = {'total_transitions', 'eval_step', 'episode', 'episode_len', 'rl_epoch', 'irl_epoch',
                                  'd_loss', 'return', 'eval_return', 'cumul_eval_return', 'avg_eval_return',
                                  'performance', 'n_wait'}

        self.use_validation = not self.discrete

        self.best_performance = -float('inf')
        self.n_wait = 1

        if self.use_validation:
            self.max_wait = 500
        else:
            self.max_wait = float('inf')

    def add_expert_path(self, expert_paths):
        self.expert_paths = expert_paths
        expert_obs, expert_act = self.expert_paths.get(('obs', 'action'))
        self.n_tot = len(expert_act)

        obs = obs_to_torch(expert_obs, device=self.device)
        act = to_torch(expert_act, device=self.device)

        if self.use_validation:
            split_prop = 0.7
            shuffle = True
        else:
            split_prop = 1.
            shuffle = False

        self.n_train = int(split_prop * self.n_tot)

        shuffling_idxs = torch.randperm(self.n_tot) if shuffle else torch.arange(0, self.n_tot)
        obs = obs.get_from_index(shuffling_idxs)
        act = act[shuffling_idxs]
        train_idx = shuffling_idxs[0:self.n_train]
        valid_idx = shuffling_idxs[self.n_train:self.n_tot]
        self.train_obs = obs.get_from_index(train_idx)
        self.train_act = act[train_idx]

        self.valid_obs = obs.get_from_index(valid_idx)
        self.valid_act = act[valid_idx]

    @torch.no_grad()
    def compute_performance(self):
        actions = self.classifier(self.valid_obs)
        error = - self.loss_function(input=actions, target=self.valid_act.long() if self.discrete else self.valid_act)
        performance = - error
        return performance.cpu().numpy()

    def fit(self, batch_size, n_epochs_per_update, **kwargs):

        if self.n_wait > self.max_wait:
            return {'d_loss': 0, 'performance': self.last_performance}

        n_itr = math.ceil(self.n_train / batch_size)
        it_epochs = TrainingIterator(max_itr=n_epochs_per_update)
        for epoch in it_epochs:
            it = TrainingIterator(max_itr=n_itr)
            shuffling_idxs = torch.randperm(self.n_train)
            self.train_obs = self.train_obs.get_from_index(shuffling_idxs)
            self.train_act = self.train_act[shuffling_idxs]
            for obs_var, targets in batch_iter(all_obs=self.train_obs,
                                               all_act=self.train_act,
                                               batch_size=batch_size):
                logits = self.classifier(obs_var)
                loss = self.loss_function(input=logits, target=targets.long() if self.discrete else targets)

                self.classifier.optim.zero_grad()
                loss.backward()
                self.classifier.optim.step()

                it.record('d_loss', to_numpy(loss))

            performance = self.compute_performance()
            if performance > self.best_performance:
                self.best_performance = performance
                self.n_wait = 0
            else:
                self.last_performance = performance
                self.n_wait += 1
            it_epochs.record('performance', performance)
            it_epochs.record('n_wait', self.n_wait)
            it_epochs.update(it.pop_all_means())

        return it_epochs.pop_all_means()

    @torch.no_grad()
    def act(self, obs, sample, **kwargs):
        obs = obs_to_torch(obs, device=self.device, unsqueeze_dim=0)

        if self.discrete:
            action = self.classifier.act(obs, sample=sample, return_log_pi=False)[0].cpu().numpy()
            action = int(action)
        else:
            action = self.classifier.act(obs)[0].cpu().numpy()

        return action

    def load_params(self, save_dict):
        self.classifier.load_state_dict(save_dict['classifier'])

    def get_params(self):
        return {'classifier': self.classifier.state_dict()}

    def prep_rollout(self, rollout_device):
        self.classifier.eval()
        to_device(self.classifier, rollout_device)

    def prep_training(self, train_device):
        self.classifier.train()
        to_device(self.classifier, train_device)

    def get_policy(self):
        return self.classifier

    @property
    def device(self):
        return self.classifier.device

    def save_training_graphs(self, train_recorder, save_dir):
        from alfred.utils.plots import plot_curves, create_fig
        import matplotlib.pyplot as plt

        # Losses

        fig, axes = create_fig((1, 1))
        plot_curves(axes,
                    ys=[train_recorder.tape['d_loss']],
                    xs=[train_recorder.tape['episode']],
                    xlabel="Episode",
                    ylabel="d_loss")

        fig.savefig(str(save_dir / 'losses.png'))
        plt.close(fig)

        # True Returns
        fig, axes = create_fig((1, 2))
        fig.suptitle('True returns')
        plot_curves(axes[0],
                    ys=[train_recorder.tape['return']],
                    xs=[train_recorder.tape['episode']],
                    xlabel="Episode", ylabel="Mean Return")
        plot_curves(axes[1],
                    ys=[train_recorder.tape['eval_return']],
                    xs=[train_recorder.tape['episode']],
                    xlabel="Episode", ylabel="Mean Eval Return")
        fig.savefig(str(save_dir / 'true_returns.png'))
        plt.close(fig)

    def wandb_watchable(self):
        return [self.classifier.network]
