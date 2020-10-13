try:
    from pommerman import constants
except ModuleNotFoundError:
    pass

import numpy as np


def one_hot_map_of_entities(state):
    # extract state variables

    board = state['board']
    pos = state['position']
    teammate = state['teammate']
    enemies = state['enemies']

    tiles = []

    # Create all tiles

    # 1) self position

    tile = np.zeros_like(board)
    tile[pos] = 1
    tiles.append(tile)

    # 2) teammates' position if we have a teammate (i.e. not AgentDummy)
    # therefore this tile is only added in the Team setting

    if not (teammate == constants.Item.AgentDummy):
        tile = np.where(board == teammate.value, 1, 0)
        tiles.append(tile)

    # 3) enemies' position for each enemy

    enemies_id = [e.value for e in enemies]
    tile = np.zeros_like(board)
    tile[np.isin(board, enemies_id)] = 1
    tiles.append(tile)

    # 4-X) positions of non-characters objects

    for object in constants.Item:
        if object not in {constants.Item.AgentDummy, constants.Item.Agent0, constants.Item.Agent1,
                          constants.Item.Agent2, constants.Item.Agent3}:
            tile = np.where(board == object.value, 1, 0)
            tiles.append(tile)
    return np.stack(tiles, axis=0).astype(np.float16)


def map_with_entities_properties(state):
    # No additional processing we just stack them
    return np.stack([state['bomb_blast_strength'],
                     state['bomb_life'],
                     state['bomb_moving_direction'],
                     state['flame_life']]).astype(np.float16)


def context_vector(state):
    return np.array([
        state['blast_strength'],
        int(state['can_kick']),
        state['ammo'],
        state['step_count'] / 1000.
    ]).astype(np.float16)
