import readchar
import imageio
import shutil


def dict_to_str(dict):
    string = ""
    for key in dict:
        string += f'{key} : {dict[key]}\n'
    return string


def wait_for_keypress(limit=1, go_key='\x1b[C'):  # default key is RIGHT_ARROW_KEY (i.e. '\x1b[C')
    i = 0
    try:
        key = readchar.readkey()
    except:
        print("Your terminal does not allow waiting")
        return

    while not (key == go_key):
        key = readchar.readkey()
        i += 1
        if i >= limit:
            exit()


def wait_for_ENTER_keypress():
    while True:
        answer = input('Press "ENTER" to continue')
        if answer == "":
            return


def save_gif_from_png_folder(folder, gif_path, fps, delete_folder=False):
    image_list = []

    def sorted_iter_dir(dir):
        return sorted(list(dir.iterdir()), key=lambda x: int(x.stem.split('_')[-1]))

    for filename in sorted_iter_dir(folder):
        if filename.is_dir():
            for f in sorted_iter_dir(filename):
                image_list.append(f)
        else:
            image_list.append(filename)

    images = [imageio.imread(f) for f in image_list]
    imageio.mimsave(str(gif_path), images, duration=1. / fps)
    if delete_folder:
        shutil.rmtree(folder)


def get_game_state(env):
    return env.get_json_info()


def load_game_states_from_demos(demos, idx):
    game_states = []

    for demo_game_states in demos['game_states']:

        if idx > len(demo_game_states) - 1:
            game_states.append(demo_game_states[len(demo_game_states) - 1])

        elif idx < 0 and abs(idx) > len(demo_game_states):
            game_states.append(demo_game_states[0])

        else:
            game_states.append(demo_game_states[idx])

    return game_states


def reset_unwrapped_env_to_init_state(unwrapped_env, init_game_state):
    unwrapped_env._init_game_state = init_game_state
    state = unwrapped_env.reset()
    return state
