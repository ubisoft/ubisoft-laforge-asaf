from utils.obs_dict import ObsDict
from alg_task_lists import TASKS, TOY_TASKS, POMMERMAN_TASKS, MUJOCO_TASKS, TOY_TASKS_CONTINUOUS, TOY_TASKS_DISCRETE
from pommerman_plugin.env_wrapper import PommermanWrapper
from utils.env_wrappers import ScaledActionsWrapper
import pommerman

import gym


def make_env(task_name):
    assert task_name in TASKS

    unwrapped_env = make_unwrapped_env(task_name)

    if task_name in TOY_TASKS_DISCRETE:
        env = VecObsDictWrapper(unwrapped_env)

    elif task_name in TOY_TASKS_CONTINUOUS:
        env = VecObsDictWrapper(unwrapped_env)
        env = ScaledActionsWrapper(env)

    elif task_name in POMMERMAN_TASKS:
        env = PommermanWrapper(unwrapped_env)

    elif task_name in MUJOCO_TASKS:
        env = VecObsDictWrapper(unwrapped_env)
        env = ScaledActionsWrapper(env)

    else:
        raise NotImplementedError
    return env


def make_unwrapped_env(task_name):
    # CLASSIC CONTROL ENVIRONMENTS

    if task_name == "mountaincar":
        return gym.make("MountainCar-v0")

    elif task_name == "cartpole":
        return gym.make("CartPole-v0")

    elif task_name == "lunarlander":
        return gym.make("LunarLander-v2")

    elif task_name == "pendulum":
        return gym.make("Pendulum-v0")

    elif task_name == "lunarlander-c":
        return gym.make("LunarLanderContinuous-v2")
    elif task_name == "mountaincar-c":
        return gym.make("MountainCarContinuous-v0")

    # MUJOCO ENVIRONMENTS

    if "hopper" in task_name:
        return gym.make("Hopper-v2")

    elif "walker2d" in task_name:
        return gym.make("Walker2d-v2")

    elif "halfcheetah" in task_name:
        return gym.make("HalfCheetah-v2")

    elif "ant" in task_name:
        return gym.make("Ant-v2")

    elif "humanoid" in task_name:
        return gym.make("Humanoid-v2")

    # POMMERMAN ENVIRONMENTS

    elif task_name == "agent47vsRandomPacifist1v1empty":
        board_size = pommerman.constants.BOARD_SIZE_ONE_VS_ONE
        agent_list = [pommerman.agents.StateAgentExploit(board_shape=(board_size, board_size)),
                      pommerman.agents.RandomPacifist()]
        return pommerman.make('OneVsOneEmpty-v0', agent_list)

    elif task_name == "learnablevsRandomPacifist1v1empty":
        agent_list = [pommerman.agents.LearnableAgent(),
                      pommerman.agents.RandomPacifist()]
        return pommerman.make('OneVsOneEmpty-v0', agent_list)

    else:
        raise NotImplementedError


class VecObsDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(VecObsDictWrapper, self).__init__(env)
        self.observation_space = {'obs_vec_size': env.observation_space}

    def observation(self, observation):
        return VecObsDictWrapper.extract_observation(observation)

    @staticmethod
    def extract_observation(state):
        return ObsDict({'obs_vec': state})


def get_observation_extractor(task_name):
    if task_name in TOY_TASKS:
        return VecObsDictWrapper.extract_observation
    elif task_name in POMMERMAN_TASKS:
        return PommermanWrapper.extract_observation
    elif task_name in MUJOCO_TASKS:
        return VecObsDictWrapper.extract_observation
    else:
        raise NotImplementedError
