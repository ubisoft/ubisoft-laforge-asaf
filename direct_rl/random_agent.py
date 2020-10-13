from base_algo import BaseAlgo
from direct_rl.models import DiscreteRandomPolicy


class RandomLearner(BaseAlgo):
    def __init__(self, action_dim, alg_name='random', **kwargs):
        assert alg_name == 'random'
        super().__init__()
        self.name = alg_name
        self.actor = DiscreteRandomPolicy(num_out=action_dim)
        self.init_dict = {'action_dim': action_dim, 'alg_name': alg_name}

        self.metrics_to_record = {'total_transitions', 'eval_step', 'episode', 'episode_len', 'epoch', 'loss', 'return',
                                  'eval_return', 'cumul_eval_return', 'avg_eval_return'}

    def train_model(self, experiences):
        return {'loss': 0}

    def get_policy(self):
        return self.actor

    def act(self, obs, sample, return_log_pi):
        return self.actor.act(obs=obs, sample=sample, return_log_pi=return_log_pi)

    def eval(self):
        pass

    def train(self):
        pass

    def get_params(self):
        return {}

    def load_params(self, save_dict):
        pass

    def wandb_watchable(self):
        return []

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
