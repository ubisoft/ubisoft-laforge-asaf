from alfred.utils.recorder import TrainingIterator

from base_algo import BaseAlgo
from utils.ml import to_torch, to_numpy, to_device, onehot_from_index, log_sum_exp, get_act_properties_from_act_space
from utils.obs_dict import obs_to_torch, torch_cat_obs
from utils.networks import MLPNetwork, CNN_MLP_hybrid
from direct_rl.models import CategoricalPolicy

import random
import torch
import torch.nn as nn
import torch.optim as optim


class AILDiscriminator(nn.Module):
    """
    Discriminator for Adversarial Imitation Learning (AIRL and ASQF)
    """

    def __init__(
            self,
            obs_space,
            act_space,
            hidden_size,
            use_advantage_formulation,
            use_multi_head):

        super(AILDiscriminator, self).__init__()

        self.obs_space = obs_space
        self.act_space = act_space
        self.action_dim, _ = get_act_properties_from_act_space(self.act_space)
        self.use_advantage_formulation = use_advantage_formulation
        self.use_multi_head = use_multi_head

        if use_multi_head:
            num_mlp_inputs = obs_space['obs_vec_size'][0]
            num_mlp_outputs = self.action_dim
        else:
            num_mlp_inputs = obs_space['obs_vec_size'][0] + self.action_dim
            num_mlp_outputs = 1

        if set(obs_space.keys()) == {'obs_vec_size'}:
            self.g = MLPNetwork(num_inputs=num_mlp_inputs,
                                num_outputs=num_mlp_outputs,
                                hidden_size=hidden_size,
                                set_final_bias=False)

        elif set(obs_space.keys()) == {'obs_map_size', 'obs_vec_size'}:
            assert obs_space['obs_map_size'][1] == obs_space['obs_map_size'][2]

            self.g = CNN_MLP_hybrid(input_vec_len=num_mlp_inputs,
                                    mlp_output_vec_len=num_mlp_outputs,
                                    mlp_hidden_size=hidden_size,
                                    input_maps_size=obs_space['obs_map_size'][1],
                                    num_input_channels=obs_space['obs_map_size'][0],
                                    cnn_output_vec_len=hidden_size,
                                    set_final_bias=False)

        else:
            raise NotImplementedError

    def get_reward(self, obs, action, log_pi_a, **kwargs):

        f_a = self.get_log_prob_from_obs_action_pairs(obs=obs, action=action)

        return f_a - log_pi_a

    def get_classification_loss(self, obs, action, log_pi_a, target):
        r"""
        Return the loss function of the discriminator to be optimized.

        As with GAN discriminator, we only want to discriminate the expert from the
        learner, thus this is a binary classification problem.

        Unlike Discriminator used for GAIL, the discriminator in this class
        takes a specific form, where

                        exp{f(s, a)}
        D(s,a) = -------------------------
                  exp{f(s, a)} + \pi(a|s)

        like for AIRL

        """

        log_numerator = self.get_log_prob_from_obs_action_pairs(obs=obs, action=action)
        to_sum = torch.cat((log_pi_a, log_numerator), 1)
        log_denominator = log_sum_exp(to_sum, 1, keepdim=True)
        # Binary Cross Entropy Loss
        # y_n = 1 for expert and 0 for generated
        # loss_n= - [y_n * log(p(x_n)) + (1 − y_n) * log(1 − p(x_n))]
        # where x_n is the input, and y_n is the label
        total_loss = (target * (log_numerator - log_denominator) + (1 - target) * (log_pi_a - log_denominator))
        loss = -total_loss.mean(0)
        return loss

    def get_log_prob_from_obs_action_pairs(self, obs, action):
        """
        Log-prob here refers to the log probability \pi_theta(a|s) of the policy learned by the discriminator
        This formulation depends on the fact that the discriminator uses the form of the optimal discriminator
        as described in the original GAN paper, combined with the form of the MaxEnt policy:

        \pi_theta(a|s) = exp{f_theta(s, a)}

        When we expand log D(s,a) with this discriminator form, we get:

                            exp{f(s, a)}
        log D(s,a) = -------------------------
                      exp{f(s, a)} + \pi_g(a|s)

                = f - log(exp{f(s,a) + \pi_g(a|s)}

                = f - log(exp{f(s,a) + exp{log \pi_g(a|s)}}

                = f - log_sum_exp(f(s,a) + log \pi_g(a|s)}

        With this derivation, we see that f(s,a) can be interpreted as the log-probability log( \pi_theta(a|s) )
        and this is why this function, which returns f(s,a) is called like this.
        """
        if self.use_multi_head:
            # action should be already be in index shape
            assert len(action.shape) == 1
            f = self.g(obs)
            f_a = f.gather(1, action.view(-1, 1))
            if self.use_advantage_formulation:
                return f_a - log_sum_exp(f, dim=1, keepdim=True)
            else:
                return f_a
        else:
            assert len(action.shape) > 1
            obs['obs_vec'] = torch.cat((obs['obs_vec'], action), -1)
            return self.g(obs)

    def get_all_log_prob_from_obs(self, obs):
        assert self.use_multi_head
        assert self.use_advantage_formulation

        f = self.g(obs)
        log_prob = f - log_sum_exp(f, dim=1, keepdim=True)
        return log_prob

    @property
    def device(self):
        return next(self.parameters()).device


class AILLearner(BaseAlgo):
    """
    Learner class for Adversarial Imitation Learning (AIRL and ASQF)
    """

    def __init__(
            self,
            obs_space,
            act_space,
            discriminator_args,
            discriminator_lr,
            alg_name,
            grad_norm_clip,
    ):

        super().__init__()

        self.obs_space = obs_space
        self.act_space = act_space
        self.action_dim, self.discrete = get_act_properties_from_act_space(act_space)
        self.discrete = bool(self.discrete)
        self.expert_var = None
        self.n_expert = None

        discriminator_args.update({'obs_space': self.obs_space, 'act_space': self.act_space})

        # Build energy model
        self.discriminator = AILDiscriminator(**discriminator_args)

        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.name = alg_name

        self.grad_norm_clip = grad_norm_clip

        self.init_dict = {'obs_space': obs_space,
                          'act_space': act_space,
                          'discriminator_args': discriminator_args,
                          'discriminator_lr': discriminator_lr,
                          'alg_name': alg_name}

        self.metrics_to_record = {'total_transitions', 'eval_step', 'episode', 'episode_len', 'irl_epoch', 'rl_epoch',
                                  'd_loss', 'return', 'eval_return', 'cumul_eval_return', 'avg_eval_return'}

    def add_expert_path(self, expert_paths):
        expert_obs, expert_act = expert_paths.get(('obs', 'action'))

        # Create the torch variables

        expert_obs_var = obs_to_torch(expert_obs, device=self.device)

        if self.discrete:
            # index is used for policy log_prob and for multi_head discriminator
            expert_act_index = expert_act.astype(int)
            expert_act_index_var = to_torch(expert_act_index, device=self.device)

            # one-hot is used with single head discriminator
            if (not self.discriminator.use_multi_head):
                expert_act = onehot_from_index(expert_act_index, self.action_dim)
                expert_act_var = to_torch(expert_act, device=self.device)

            else:
                expert_act_var = expert_act_index_var
        else:
            # there is no index actions for continuous control so index action and normal actions are the same variable
            expert_act_var = to_torch(expert_act, device=self.device)

            expert_act_index_var = expert_act_var

        self.expert_var = (expert_obs_var, expert_act_var)
        self.expert_act_index_var = expert_act_index_var
        self.n_expert = len(expert_obs)

    def fit(self, data, batch_size, policy, n_epochs_per_update, logger, **kwargs):
        """
        Train the Discriminator to distinguish expert from learner.
        """
        obs, act = data[0], data[1]

        # Create the torch variables

        obs_var = obs_to_torch(obs, device=self.device)

        if self.discrete:
            # index is used for policy log_prob and for multi_head discriminator
            act_index = act.astype(int)
            act_index_var = to_torch(act_index, device=self.device)

            # one-hot is used with single head discriminator
            if (not self.discriminator.use_multi_head):
                act = onehot_from_index(act_index, self.action_dim)
                act_var = to_torch(act, device=self.device)

            else:
                act_var = act_index_var
        else:
            # there is no index actions for continuous control index so action and normal actions are the same variable
            act_var = to_torch(act, device=self.device)
            act_index_var = act_var

        expert_obs_var, expert_act_var = self.expert_var
        expert_act_index_var = self.expert_act_index_var

        # Eval the prob of the transition under current policy
        # The result will be fill in part to the discriminator, no grad because if policy is discriminator as for ASQF
        # we do not want gradient passing
        with torch.no_grad():
            trans_log_probas = policy.get_log_prob_from_obs_action_pairs(obs=obs_var, action=act_index_var)
            expert_log_probas = policy.get_log_prob_from_obs_action_pairs(obs=expert_obs_var,
                                                                          action=expert_act_index_var)

        n_trans = len(obs)
        n_expert = self.n_expert

        # Train discriminator
        for it_update in TrainingIterator(n_epochs_per_update):
            shuffled_idxs_trans = torch.randperm(n_trans, device=self.device)

            for i, it_batch in enumerate(TrainingIterator(n_trans // batch_size)):

                # the epoch is defined on the collected transition data and not on the expert data

                batch_idxs_trans = shuffled_idxs_trans[batch_size * i: batch_size * (i + 1)]
                batch_idxs_expert = torch.tensor(random.sample(range(n_expert), k=batch_size), device=self.device)

                # lprobs_batch is the prob of obs and act under current policy

                obs_batch = obs_var.get_from_index(batch_idxs_trans)
                act_batch = act_var[batch_idxs_trans]
                lprobs_batch = trans_log_probas[batch_idxs_trans]

                # expert_lprobs_batch is the experts' obs and act under current policy

                expert_obs_batch = expert_obs_var.get_from_index(batch_idxs_expert)
                expert_act_batch = expert_act_var[batch_idxs_expert]
                expert_lprobs_batch = expert_log_probas[batch_idxs_expert]

                labels = torch.zeros((batch_size * 2, 1), device=self.device)
                labels[batch_size:] = 1.0  # expert is one
                total_obs_batch = torch_cat_obs([obs_batch, expert_obs_batch], dim=0)
                total_act_batch = torch.cat([act_batch, expert_act_batch], dim=0)

                total_lprobs_batch = torch.cat([lprobs_batch, expert_lprobs_batch], dim=0)

                loss = self.discriminator.get_classification_loss(obs=total_obs_batch, action=total_act_batch,
                                                                  log_pi_a=total_lprobs_batch, target=labels)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_norm_clip)
                self.optimizer.step()

                it_update.record('d_loss', loss.cpu().data.numpy())

        return_dict = {}

        return_dict.update(it_update.pop_all_means())

        return return_dict

    def get_reward(self, obs, action, log_pi_a, **kwargs):
        return self.discriminator.get_reward(obs=obs, log_pi_a=log_pi_a, action=action)

    def update_reward(self, experiences, policy, ent_wt):
        (obs, action, next_obs, g_reward, mask) = experiences
        obs_var = obs_to_torch(obs, device=self.device)
        if self.discrete:
            if (not self.discriminator.use_multi_head):
                action_idx = action.astype(int)
                action = onehot_from_index(action_idx, self.action_dim)
            else:
                action_idx = action.astype(int)
                action = action_idx
        else:
            action_idx = action

        log_pi_list = policy.get_log_prob_from_obs_action_pairs(action=to_torch(action_idx, device=self.device),
                                                                obs=obs_var).detach()
        reward = to_numpy(self.get_reward(obs=obs_var,
                                          action=to_torch(action, device=self.device),
                                          log_pi_a=log_pi_list,
                                          ent_wt=ent_wt).squeeze().detach())

        return (obs, action_idx, next_obs, reward, mask)

    def act(self, obs, sample):
        assert self.discriminator.use_multi_head
        #  the softmax makes no difference if from Q or advantages
        logits = self.discriminator.h(obs_to_torch(obs, unsqueeze_dim=0, device=self.device))
        return int(
            CategoricalPolicy.act_from_logits(logits=logits, sample=sample, return_log_pi=False).cpu().numpy())

    def get_policy(self):
        assert self.discriminator.use_multi_head
        return self.discriminator

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
        return [self.discriminator.h]
