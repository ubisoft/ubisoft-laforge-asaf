import numpy as np

# IMPORTANT
# these functions are based on how observation maps are constructed in observation_extractor.py
# the following indices are assumed for the obs_map:
objType_to_idx = {
    'self-position': 0,
    'enemies': 1,
    'Passage': 2,
    'Rigid': 3,
    'Wood': 4,
    'Bomb': 5,
    'Flames': 6,
    'Fog': 7,
    'ExtraBomb': 8,
    'IncrRange': 9,
    'Kick': 10,
    'bomb_blast_strength': 11,
    'bomb_life': 12,
    'bomb_moving_direction': 13,
    'flame_life': 14
}


def get_n_enemies(obs_map):
    return np.sum(obs_map[objType_to_idx['enemies']])


def get_n_woods(obs_map):
    return np.sum(obs_map[objType_to_idx['Wood']])


def get_n_powerUps(obs_map):
    return np.sum(obs_map[objType_to_idx['ExtraBomb']]) + \
           np.sum(obs_map[objType_to_idx['IncrRange']]) + \
           np.sum(obs_map[objType_to_idx['Kick']])
