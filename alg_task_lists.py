RL_OFF_POLICY_ALGS = {"sac", "sacmh", "sqil", "sqil-c", "random"}
# note that sqil is an IRL algo but is considered as an RL algo due to its implementation

RL_ON_POLICY_ALGS = {"ppo", "random"}

RL_ALGS = RL_OFF_POLICY_ALGS | RL_ON_POLICY_ALGS

IRL_ALGS = {"asaf-fullX", "asaf-wX", "asaf-1X", "asqfX", "bcX", "airlXppo", "gailXppo"}

ALGS = RL_ALGS | IRL_ALGS

POMMERMAN_UNWRAPPED_TASKS = {
                             'agent47vsRandomPacifist1v1empty',
                             }

POMMERMAN_TASKS = {
                   'learnablevsRandomPacifist1v1empty'
                   }

TOY_TASKS_DISCRETE = {"mountaincar",
                      "cartpole",
                      "lunarlander"}

TOY_TASKS_CONTINUOUS = {'pendulum',
                        'lunarlander-c',
                        'mountaincar-c'}

MUJOCO_TASKS = {"hopper-c", "walker2d-c", "halfcheetah-c", "ant-c"}

TOY_TASKS = TOY_TASKS_DISCRETE | TOY_TASKS_CONTINUOUS
TASKS = POMMERMAN_TASKS | TOY_TASKS | MUJOCO_TASKS
