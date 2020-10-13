from alfred.utils.recorder import TrainingIterator

from base_algo import BaseAlgo
from utils.ml import to_torch, to_numpy, to_device, onehot_from_index, get_act_properties_from_act_space
from utils.obs_dict import obs_to_torch, torch_cat_obs, ObsDict
from utils.networks import MLPNetwork, CNN_MLP_hybrid

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import Box, Discrete


class GAILDiscriminator(nn.Module):
    """
    Discriminator function for Generative Adversarial Imitation Learning
    see: https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf
    """

    def __init__(self, obs_space, act_space, hidden_size):
        super(GAILDiscriminator, self).__init__()

        self.obs_space = obs_space
        self.act_space = act_space

        self.action_dim, _ = get_act_properties_from_act_space(self.act_space)

        num_mlp_inputs = obs_space['obs_vec_size'][0] + self.action_dim
        num_mlp_outputs = 1

        if set(obs_space.keys()) == {'obs_vec_size'}:
            self.d = MLPNetwork(num_inputs=num_mlp_inputs,
                                num_outputs=num_mlp_outputs,
                                hidden_size=hidden_size,
                                set_final_bias=False)

        elif set(obs_space.keys()) == {'obs_map_size', 'obs_vec_size'}:
            assert obs_space['obs_map_size'][1] == obs_space['obs_map_size'][2]

            self.d = CNN_MLP_hybrid(input_vec_len=num_mlp_inputs,
                                    mlp_output_vec_len=num_mlp_outputs,
                                    mlp_hidden_size=hidden_size,
                                    input_maps_size=obs_space['obs_map_size'][1],
                                    num_input_channels=obs_space['obs_map_size'][0],
                                    cnn_output_vec_len=hidden_size,
                                    set_final_bias=False)

        else:
            raise NotImplementedError

    def get_reward(self, obs, action, log_pi_a, ent_wt, definition):
        """"
        negative reward definition:  log D
        positive reward definition: - log(1-D)
        """
        logits = self.forward_prop_s_a(obs=obs, action=action)

        if definition == "negative":
            reward = nn.functional.logsigmoid(logits)
        elif definition == "positive":  # note: 1 - sigmoid(x) = sigmoid(-x)
            reward = - nn.functional.logsigmoid(-logits)
        else:
            raise NotImplementedError(f'd_reward_definition : {definition} is not a correct argument')

        return reward - ent_wt * log_pi_a  # reward + entropy (both to be maximised by RL step)

    def get_classification_loss(self, obs, action, target):
        bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        logits = self.forward_prop_s_a(obs=obs, action=action)
        return bce_loss(logits, target)

    def forward_prop_s_a(self, obs, action):
        assert len(action.shape) > 1
        obs['obs_vec'] = torch.cat((obs['obs_vec'], action), -1)
        return self.d(obs)

    def get_grad_penality(self, obs_e, act_e, obs_l, act_l, gradient_penalty_coef=10):
        batch_size = len(act_e)
        alpha = torch.rand(batch_size, 1)

        data_e = {key: obs_e[key] for key in obs_e.keys()}
        data_e['obs_vec'] = torch.cat([obs_e['obs_vec'], act_e], dim=1)

        data_l = {key: obs_l[key] for key in obs_l.keys()}
        data_l['obs_vec'] = torch.cat([obs_l['obs_vec'], act_l], dim=1)

        mixup_data = {}
        for key in data_e.keys():
            alpha_temp = alpha.expand_as(data_e[key]).to(data_e[key].device)
            mixup_data[key] = alpha_temp * data_e[key] + (1 - alpha_temp) * data_l[key]
            mixup_data[key].requires_grad = True

        disc = self.d(ObsDict(mixup_data))

        ones = torch.ones(disc.size()).to(disc.device)

        grad = torch.autograd.grad(outputs=disc,
                                   inputs=list(mixup_data.values()),
                                   grad_outputs=ones,
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)

        grad = torch.cat([g.view(batch_size, -1) for g in grad])

        grad_pen = gradient_penalty_coef * (grad.norm(2, dim=1) - 1).pow(2).mean(0)
        return grad_pen

    @property
    def device(self):
        return next(self.parameters()).device


class GAILLearner(BaseAlgo):
    """
    Learner class for Generative Adversarial Imitation Learning
    """

    def __init__(
            self,
            obs_space,
            act_space,
            discriminator_args,
            discriminator_lr,
            reward_definition,
            grad_norm_clip,
            gradient_penalty_coef,
            alg_name,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        discriminator_args.update({'obs_space': self.obs_space, 'act_space': self.act_space})
        self.grad_norm_clip = grad_norm_clip
        self.gradient_penalty_coef = gradient_penalty_coef
        self.data_e = None
        self.n_expert = None

        # Build model and optimiser
        self.discriminator = GAILDiscriminator(**discriminator_args)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.name = alg_name

        self.reward_definition = reward_definition

        self.init_dict = {'obs_space': obs_space,
                          'act_space': act_space,
                          'discriminator_args': discriminator_args,
                          'discriminator_lr': discriminator_lr,
                          'alg_name': alg_name,
                          'reward_definition': reward_definition}

        self.metrics_to_record = {'total_transitions', 'eval_step', 'episode', 'episode_len', 'irl_epoch', 'rl_epoch',
                                  'd_loss'}

    def add_expert_path(self, expert_paths):
        expert_obs, expert_act = expert_paths.get(('obs', 'action'))

        if isinstance(self.act_space, Discrete):
            # convert actions from integer representation to one-hot representation

            expert_act = onehot_from_index(expert_act.astype(int), self.action_dim)

        # create the torch variables
        self.data_e = (
            obs_to_torch(expert_obs, device=self.device),
            to_torch(expert_act, device=self.device),
        )
        self.n_expert = len(expert_obs)

    def fit(self, data, batch_size, n_epochs_per_update, logger, **kwargs):
        """
        Train the discriminator to distinguish expert from learner.
        """
        agent_obs, agent_act = data[0], data[1]

        if isinstance(self.act_space, Discrete):
            # convert actions from integer representation to one-hot representation

            agent_act = onehot_from_index(agent_act.astype(int), self.action_dim)

        assert self.data_e is not None
        assert self.n_expert is not None

        # create the torch variables

        agent_obs_var = obs_to_torch(agent_obs, device=self.device)
        expert_obs_var = self.data_e[0]

        act_var = to_torch(agent_act, device=self.device)
        expert_act_var = self.data_e[1]

        # Train discriminator for n_epochs_per_update

        n_trans = len(agent_obs)
        n_expert = self.n_expert

        for it_update in TrainingIterator(n_epochs_per_update):  # epoch loop

            shuffled_idxs_trans = torch.randperm(n_trans, device=self.device)
            for i, it_batch in enumerate(TrainingIterator(n_trans // batch_size)):  # mini-bathc loop

                # the epoch is defined on the collected transition data and not on the expert data

                batch_idxs_trans = shuffled_idxs_trans[batch_size * i: batch_size * (i + 1)]
                batch_idxs_expert = torch.tensor(random.sample(range(n_expert), k=batch_size), device=self.device)

                # get mini-batch of agent transitions

                obs_batch = agent_obs_var.get_from_index(batch_idxs_trans)
                act_batch = act_var[batch_idxs_trans]

                # get mini-batch of expert transitions

                expert_obs_batch = expert_obs_var.get_from_index(batch_idxs_expert)
                expert_act_batch = expert_act_var[batch_idxs_expert]

                labels = torch.zeros((batch_size * 2, 1), device=self.device)
                labels[batch_size:] = 1.0  # expert is one
                total_obs_batch = torch_cat_obs([obs_batch, expert_obs_batch], dim=0)
                total_act_batch = torch.cat([act_batch, expert_act_batch], dim=0)

                loss = self.discriminator.get_classification_loss(obs=total_obs_batch, action=total_act_batch,
                                                                           target=labels)

                if self.gradient_penalty_coef != 0.0:
                    grad_penalty = self.discriminator.get_grad_penality(
                        obs_e=expert_obs_batch, obs_l=obs_batch, act_e=expert_act_batch, act_l=act_batch,
                        gradient_penalty_coef=self.gradient_penalty_coef
                    )
                    loss += grad_penalty

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_norm_clip)
                self.optimizer.step()

                it_update.record('d_loss', loss.cpu().data.numpy())

        return_dict = {}
        return_dict.update(it_update.pop_all_means())

        return return_dict

    def get_reward(self, obs, action, log_pi_a, ent_wt):
        return self.discriminator.get_reward(obs=obs, log_pi_a=log_pi_a,
                                             ent_wt=ent_wt, action=action,
                                             definition=self.reward_definition)

    def update_reward(self, experiences, policy, ent_wt):
        (obs, action, next_obs, g_reward, mask) = experiences
        obs_var = obs_to_torch(obs, device=self.device)

        if isinstance(self.act_space, Discrete):
            # convert actions from integer representation to one-hot representation
            action_idx = action.astype(int)
            action = onehot_from_index(action_idx, self.action_dim)
        else:
            action_idx = action

        log_pi_list = policy.get_log_prob_from_obs_action_pairs(action=to_torch(action_idx, device=self.device),
                                                                obs=obs_var).detach()
        reward = to_numpy(self.get_reward(obs=obs_var,
                                          action=to_torch(action, device=self.device),
                                          log_pi_a=log_pi_list,
                                          ent_wt=ent_wt).squeeze().detach())

        return (obs, action_idx, next_obs, reward, mask)

    def load_params(self, ckpt):
        self.discriminator.load_state_dict(ckpt['discriminator'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def get_params(self):
        return {'discriminator': self.discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def prep_rollout(self, rollout_device):
        self.discriminator.eval()
        to_device(self.discriminator, rollout_device)

    def prep_training(self, train_device):
        self.discriminator.train()
        to_device(self.discriminator, train_device)

    @property
    def action_dim(self):
        return self.discriminator.action_dim

    @property
    def device(self):
        return self.discriminator.device

    def save_training_graphs(self, train_recorder, save_dir):
        from alfred.utils.plots import create_fig, plot_curves
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

        # Accuracies
        to_plot = ('recall', 'specificity', 'precision', 'accuracy', 'F1')
        if any([k in train_recorder.tape for k in to_plot]):
            fig, axes = create_fig((1, 1))
            ys = [train_recorder.tape[key] for key in to_plot]
            plot_curves(axes, ys=ys,
                        xs=[train_recorder.tape['episode']] * len(ys),
                        xlabel='Episode',
                        ylabel='-',
                        labels=to_plot)
            fig.savefig(str(save_dir / 'Accuracy.png'))
            plt.close(fig)

    def wandb_watchable(self):
        return [self.discriminator.d]
