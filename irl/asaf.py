from alfred.utils.recorder import TrainingIterator

from base_algo import BaseAlgo
from utils.ml import to_device, get_act_properties_from_act_space
from utils.obs_dict import obs_to_torch, torch_cat_obs, to_torch
from direct_rl.models import GaussianPolicy, CategoricalPolicy
import numpy as np
import torch
import torch.nn as nn
import itertools


class ASAFDiscriminator(nn.Module):

    def __init__(
            self,
            obs_space,
            act_space,
            hidden_size,
            lr):

        super().__init__()

        self.obs_space = obs_space
        self.act_space = act_space

        _, is_discrete = get_act_properties_from_act_space(self.act_space)

        if is_discrete:
            self.pi = CategoricalPolicy(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr,
                                        set_final_bias=False)
        else:
            self.pi = GaussianPolicy(obs_space=obs_space, act_space=act_space, hidden_size=hidden_size, lr=lr,
                                     action_squashing='none', set_final_bias=True)

    @property
    def device(self):
        return next(self.parameters()).device


class ASAFLearner(BaseAlgo):

    def __init__(
            self,
            obs_space,
            act_space,
            discriminator_args,
            break_traj_to_windows,
            window_size,
            window_stride,
            window_over_episode,
            grad_value_clip,
            grad_norm_clip,
            alg_name
    ):

        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.break_traj_to_windows = break_traj_to_windows
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_over_episode = window_over_episode

        self.data_e = None

        discriminator_args.update({'obs_space': self.obs_space, 'act_space': self.act_space})

        # Build energy model
        self.discriminator = ASAFDiscriminator(**discriminator_args)
        self.grad_value_clip = grad_value_clip
        self.grad_norm_clip = grad_norm_clip

        self.name = alg_name

        self.init_dict = {'obs_space': obs_space,
                          'act_space': act_space,
                          'discriminator_args': discriminator_args,
                          'break_traj_to_windows': break_traj_to_windows,
                          'window_size': window_size,
                          'window_stride': window_stride,
                          'window_over_episode': window_over_episode,
                          'grad_value_clip': grad_value_clip,
                          'grad_norm_clip': grad_norm_clip,
                          'alg_name': alg_name}

        self.metrics_to_record = {'total_transitions', 'eval_step', 'episode', 'episode_len',
                                  'rl_epoch', 'irl_epoch', 'return', 'eval_return',
                                  'cumul_eval_return', 'avg_eval_return',
                                  'd_loss', 'd_loss_max', 'd_loss_min', 'd_loss_std'}

        for record in ['mb_old_adv_l', 'mb_old_adv_e', 'mb_new_adv_l', 'mb_new_adv_e']:
            for k in ['mean', 'max', 'min', 'std']:
                self.metrics_to_record.add(f'{record}_{k}')

    def add_expert_path(self, expert_paths):
        obs, act, mask = expert_paths.get(['obs', 'action', 'mask'])
        self.data_e = (
            obs_to_torch(obs, device=self.device),
            to_torch(act, device=self.device),
            mask,
        )

    def mask_to_window_start(self, done_mask):
        # If disregarding episodes, one window starts at every 'window_stride' transitions
        # (done_mask is a list of bool indicating when episodes are NOT terminated i.e. inverse of done)
        window_starts = np.zeros((len(done_mask),))
        window_starts[::self.window_stride] = 1

        if self.window_size > 1:
            # Makes sure that the last transitions are included in a last window (may not respect the stride pattern)
            window_starts[- self.window_size + 1:] = 0
            window_starts[- self.window_size] = 1

        # Returns a list of indices at which each window should start w.r.t the ordered transition buffer
        idx_to_start_from = np.where(window_starts == 1)[0]
        return idx_to_start_from

    def mask_to_episodes_limits(self, done_mask):
        # Locates the episode starting points and returns a list of such indices w.r.t the ordered transition buffer
        # (done_mask is a list of bool indicating when episodes are NOT terminated i.e. inverse of done)
        episode_ends = np.where(np.asarray(done_mask) == 0)[0]
        episodes_starts = np.concatenate((np.array([0]), episode_ends[:-1] + 1))
        return episodes_starts, episode_ends

    def split_episodes_to_windows(self, ep_start, ep_end):
        assert self.window_size == self.window_stride, "Stride is only implemented for window_over_episode to True"

        windows_start = []
        windows_end = []
        for start, end in zip(ep_start, ep_end):

            ep_len = end - start + 1
            n_fixed_len_windows = ep_len // self.window_size
            len_last_window = ep_len % self.window_size
            n_windows = n_fixed_len_windows + int(not len_last_window == 0)

            # we add a new starting point for every window that can be inserted in the episode
            windows_start += [start + i*self.window_size for i in range(n_windows)]

            # we add a new end point for every FIXED size window that can be inserted in the episode
            windows_end += [start + (i + 1) * self.window_size - 1 for i in range(n_fixed_len_windows)]

            # adds the last end point if it has not been added yet (i.e. size is smaller than normal window)
            windows_end += [end] if len_last_window > 0 else []

        return windows_start, windows_end

    def fit(self, data, batch_size, n_epochs_per_update, logger, **kwargs):

        # GET OBSERVATIONS / ACTIONS FOR LEARNER (l) AND EXPERT (e)

        obs_l, act_l, _, _, mask_l = data
        obs_e, act_e, mask_e = self.data_e

        if self.break_traj_to_windows:
            if self.window_over_episode:
                window_start_l = self.mask_to_window_start(mask_l)
                window_start_e = self.mask_to_window_start(mask_e)
                window_idx_l = [list(range(s, s + self.window_size)) for s in window_start_l]
                window_idx_e = [list(range(s, s + self.window_size)) for s in window_start_e]
                batch_size_e = batch_size
                batch_size_l = batch_size
            else:
                ep_start_l, ep_end_l = self.mask_to_episodes_limits(mask_l)
                window_start_l, window_end_l = self.split_episodes_to_windows(ep_start_l, ep_end_l)
                ep_start_e, ep_end_e = self.mask_to_episodes_limits(mask_e)
                window_start_e, window_end_e = self.split_episodes_to_windows(ep_start_e, ep_end_e)
                window_idx_l = [list(range(s, e + 1)) for s, e in zip(window_start_l, window_end_l)]
                window_idx_e = [list(range(s, e + 1)) for s, e in zip(window_start_e, window_end_e)]
                batch_size_l = int(len(window_start_l) / (kwargs['config'].d_episodes_between_updates/batch_size))
                batch_size_e = int(len(window_start_e) / (kwargs['config'].d_episodes_between_updates / batch_size))
                # we increase the batch_size so that even if the episode length increases the number of steps per epoch
                # remains constant. This is because for asaf-w on episodes (cannot window across episodes) the batch_size
                # is defined in terms of trajectories and not in terms of windows. Also note that this is not used for
                # asaf-w on transitions (can window across episodes) since the episode length does not influences the
                # optimization anymore.
        else:
            window_start_l, window_end_l = self.mask_to_episodes_limits(mask_l)
            window_start_e, window_end_e = self.mask_to_episodes_limits(mask_e)
            window_idx_l = [list(range(s, e + 1)) for s, e in zip(window_start_l, window_end_l)]
            window_idx_e = [list(range(s, e + 1)) for s, e in zip(window_start_e, window_end_e)]
            batch_size_e = batch_size
            batch_size_l = batch_size

        n_windows_l, n_windows_e = len(window_start_l), len(window_start_e)

        # COMPUTE ADVANTAGES USING (OLD) POLICY

        obs_l = obs_to_torch(obs_l, device=self.device)
        act_l = to_torch(act_l, device=self.device)

        with torch.no_grad():
            old_adv_l = self.discriminator.pi.get_log_prob_from_obs_action_pairs(
                obs=obs_l, action=act_l).squeeze()
            old_adv_e = self.discriminator.pi.get_log_prob_from_obs_action_pairs(
                obs=obs_e, action=act_e).squeeze()

        # TRAIN DISCRIMINATOR

        for it_update in TrainingIterator(n_epochs_per_update):
            shuffled_window_idx_l = np.random.permutation(n_windows_l)

            for i in range(n_windows_l // batch_size_l):

                # Shuffle the window indexes

                mb_rand_window_num_l = shuffled_window_idx_l[i * batch_size_l:(i + 1) * batch_size_l]
                mb_rand_window_num_e = np.random.randint(low=0, high=n_windows_e, size=batch_size_e)  # expert sample idx are sampled during iteration.

                # Get the indices corresponding window and flatten into one long list

                mb_window_idx_l = [window_idx_l[num] for num in mb_rand_window_num_l]
                mb_window_idx_e = [window_idx_e[num] for num in mb_rand_window_num_e]

                mb_flat_idx_l = list(itertools.chain(*mb_window_idx_l))
                mb_flat_idx_e = list(itertools.chain(*mb_window_idx_e))

                # Count window lengths

                mb_window_len_l = [len(window) for window in mb_window_idx_l]
                mb_window_len_e = [len(window) for window in mb_window_idx_e]

                # Concatenate collected and expert data

                mb_obs = torch_cat_obs((obs_l.get_from_index(mb_flat_idx_l), obs_e.get_from_index(mb_flat_idx_e)), dim=0)
                mb_act = torch.cat((act_l[mb_flat_idx_l], act_e[mb_flat_idx_e]), dim=0)
                mb_window_len = mb_window_len_l + mb_window_len_e

                # Compute (new) advantages using (current) policy and concatenate old advantages

                mb_new_adv = self.discriminator.pi.get_log_prob_from_obs_action_pairs(obs=mb_obs, action=mb_act)
                mb_old_adv = torch.cat((old_adv_l[mb_flat_idx_l], old_adv_e[mb_flat_idx_e]), dim=0).view(-1, 1)

                # Sum the advantages "window-wise" to be left with a vector of window-sums (length=2*mb_size)
                # This implements the sum of exponents in discriminator: q_pi = \prod_t exp(adv_t) = exp(\sum_t adv_t)

                new_sum_adv = torch.stack(
                    [s.sum(0) for s in torch.split(mb_new_adv, mb_window_len)],
                    dim=0).view(-1, 1)
                old_sum_adv = torch.stack(
                    [s.sum(0) for s in torch.split(mb_old_adv, mb_window_len)],
                    dim=0).view(-1, 1)

                # Computes the structured discriminator's binary cross-entropy loss
                # prob for agent:  log D = log-numerator - log-denominator = adv_pi - log ( exp(adv_pi) + exp(adv_g) )
                # prob for expert:  log D = log-numerator - log-denominator = adv_g - log ( exp(adv_pi) + exp(adv_g) )

                to_sum = torch.cat((new_sum_adv, old_sum_adv), dim=1)
                log_denominator = to_sum.logsumexp(dim=1, keepdim=True)
                target = torch.zeros_like(log_denominator)
                target[batch_size_l:] = 1  # second half of the data is from the expert

                loss = - (target * (new_sum_adv - log_denominator) + (1 - target) * (
                        old_sum_adv - log_denominator)).mean(0)

                # Backpropagation and gradient step

                self.discriminator.pi.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.discriminator.pi.parameters(), self.grad_value_clip)
                torch.nn.utils.clip_grad_norm_(self.discriminator.pi.parameters(), self.grad_norm_clip)
                self.discriminator.pi.optim.step()

                # Book-keeping

                it_update.record('d_loss', loss.cpu().data.numpy())

        vals = it_update.pop('d_loss')

        return_dict = {
            'd_loss': np.mean(vals),
            'd_loss_max': np.max(vals),
            'd_loss_min': np.min(vals),
            'd_loss_std': np.std(vals)
        }

        return return_dict

    def split_traj_len_list_to_windows(self, traj_len_list):
        windowed_traj_len_list = []
        for traj_len in traj_len_list:
            n_windows = traj_len // self.window_size

            if n_windows > 0:
                size_of_first_window = self.window_size
                remaining_len_of_traj = traj_len - size_of_first_window
                n_fixed_len_windows = remaining_len_of_traj // self.window_size
                size_of_last_window = remaining_len_of_traj % self.window_size
                windowed_traj_len_list += [size_of_first_window] + \
                                          [self.window_size for _ in range(n_fixed_len_windows)] + \
                                          [size_of_last_window]
            else:
                windowed_traj_len_list += [traj_len]
        return windowed_traj_len_list

    @torch.no_grad()
    def act(self, obs, sample):
        obs = obs_to_torch(obs, device=self.device, unsqueeze_dim=0)
        act = self.discriminator.pi.act(obs, sample, return_log_pi=False).detach()
        return act[0].cpu().numpy()

    def get_policy(self):
        return self.discriminator.pi

    def load_params(self, ckpt):
        self.discriminator.load_state_dict(ckpt['discriminator'])
        self.discriminator.pi.optim.load_state_dict(ckpt['optimizer'])

    def get_params(self):
        return {'discriminator': self.discriminator.state_dict(),
                'optimizer': self.discriminator.pi.optim.state_dict()}

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

        # Estimated Returns
        to_plot = ('IRLAverageEntReward',
                   'IRLAverageF',
                   'IRLAverageLogPi',
                   'IRLMedianLogPi',
                   'ExpertIRLAverageEntReward',
                   'ExpertIRLAverageF',
                   'ExpertIRLAverageLogPi',
                   'ExpertIRLMedianLogPi')
        if any([k in train_recorder.tape for k in to_plot]):
            fig, axes = create_fig((2, 5))
            for i, key in enumerate(to_plot):
                if key in train_recorder.tape:
                    ax = axes[i // 5, i % 5]
                    plot_curves(ax, ys=[train_recorder.tape[key]],
                                xs=[train_recorder.tape['episode']],
                                xlabel='Episode',
                                ylabel=key)
            fig.savefig(str(save_dir / 'estimated_rewards.png'))
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
        return [self.discriminator.pi.network]
