from pommerman_plugin.observation_extractor import *
from pommerman_plugin.info_extractor import get_n_woods, get_n_enemies, get_n_powerUps
from pommerman_plugin.misc import reset_unwrapped_env_to_init_state
from utils.obs_dict import ObsDict

import gym


class PommermanWrapper(gym.Wrapper):
    def __init__(self, env, learner_id=0, init_game_states=None):
        super(PommermanWrapper, self).__init__(env)
        self.init_game_states = init_game_states
        self.states = None
        self.learner_id = learner_id
        tmp_obs = self.reset()

        # Here upper bound of Boxes are arbitrary
        obs_map_size = gym.spaces.Box(low=0.0, high=10.0,
                                      shape=tmp_obs['obs_map'].shape,
                                      dtype=tmp_obs['obs_map'].dtype)
        obs_vec_size = gym.spaces.Box(low=0.0, high=10.0,
                                      shape=tmp_obs['obs_vec'].shape,
                                      dtype=tmp_obs['obs_vec'].dtype)

        self.observation_space = {'obs_vec_size': obs_vec_size,
                                  'obs_map_size': obs_map_size}

    def observation(self, observation):
        return PommermanWrapper.extract_observation(observation)

    def reset(self):
        if self.init_game_states is None:
            self.states = reset_unwrapped_env_to_init_state(unwrapped_env=self.unwrapped,
                                                            init_game_state=None)
        else:
            self.states = reset_unwrapped_env_to_init_state(unwrapped_env=self.unwrapped,
                                                            init_game_state=np.random.choice(self.init_game_states))

        return self.observation(self.states[self.learner_id])

    def step(self, action):
        all_agents_actions = self.env.act(self.states)
        all_agents_actions[self.learner_id] = action  # overwrites learnable agent's action
        states, reward, done, info = super(PommermanWrapper, self).step(all_agents_actions)
        self.states = states

        obs = self.observation(self.states[self.learner_id])
        rew = reward[self.learner_id]

        ## if the agent we are following is dead (in FFA) we overide the done to stop the episode, this might cause
        ## problem in direct_rl because it is like defining it as an absorbing states with 0 value (whereas the actual
        ## value should be -1) but does not poses a problem for IL with ASAF.
        ## this has no effect in a Team setting
        if reward[self.learner_id] == -1:
            done = True

        ## adds extra information about the maps
        info['n_woods'] = get_n_woods(obs_map=obs['obs_map'])
        info['n_enemies'] = get_n_enemies(obs_map=obs['obs_map'])
        info['n_powerUps'] = get_n_powerUps(obs_map=obs['obs_map'])

        return obs, rew, done, info

    @staticmethod
    def extract_observation(state):
        obs_map = np.concatenate([one_hot_map_of_entities(state), map_with_entities_properties(state)])
        return ObsDict({'obs_map': obs_map,
                        'obs_vec': context_vector(state)})
