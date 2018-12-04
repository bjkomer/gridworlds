# Interact with an environment using the keyboard
# NOTE: this code seems to require gym version 0.7.4
import sys
import os
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.maze_generation import generate_maze
from gridworlds.constants import possible_objects
import numpy as np
import time
import json
import argparse
from collections import OrderedDict
# from map_layout import map_layouts
from gym_maze.envs.generators import SimpleMazeGenerator, RandomMazeGenerator, \
        RandomBlockMazeGenerator, TMazeGenerator, WaterMazeGenerator
try:
    from state_prediction_utils import generate_classifier_function, RecurrentClassifier
except:
    print("Could not import state_prediction_utils, 'view_ghost' will not work")

import tensorflow as tf
# check here if baselines doesn't install properly:
#     https://github.com/openai/baselines/issues/114
#     https://ubuntuforums.org/showthread.php?t=1375500
from baselines.ppo1.mlp_policy import MlpPolicy
import os.path as osp

from gridworlds.getch import getch

# This will parse the command line arguments
from arguments import add_map_arguments, args_to_dict

parser = argparse.ArgumentParser('Interactively explore an environment')

parser = add_map_arguments(parser)

parser.add_argument('--view-ghost', action='store_true', help='view classifier prediction. Model chosen based on clf- arguments')
parser.add_argument('--clf-folder', type=str, default='sample_points/1535138810.1322715')
parser.add_argument('--clf-epochs', type=int, default=1000)
parser.add_argument('--clf-dropout', type=float, default=0.4)
parser.add_argument('--clf-network-string', type=str, default='128-128')
parser.add_argument('--clf-features', type=str, default='hd-bc-dists')
parser.add_argument('--clf-normalize-output', action='store_true')
parser.add_argument('--clf-predict-heading', action='store_true')
parser.add_argument('--clf-history', type=int, default=1, help='length of observation history to use for prediction')
parser.add_argument('--goal-distance', type=int, default=0, help='distance of the goal from the start location')

args = parser.parse_args()
params = args_to_dict(args)
obs_dict = generate_obs_dict(params)

if params['view_ghost']:
    print("Overwritting parameters with sample point file")
    folder = params['clf_folder']  # "sample_points/1535138810.1322715"

    if params['clf_history'] == 1:
        classifier = generate_classifier_function(
            folder=folder,
            epochs=params['clf_epochs'],
            dropout=params['clf_dropout'],
            network_string=params['clf_network_string'],
            features=params['clf_features'],  # 'hd-bc-dists',
            normalize_output=params['clf_normalize_output'],
            predict_heading=params['clf_predict_heading'],
        )
    else:
        classifier = RecurrentClassifier(
            folder=folder,
            epochs=params['clf_epochs'],
            dropout=params['clf_dropout'],
            network_string=params['clf_network_string'],
            features=params['clf_features'],  # 'hd-bc-dists',
            normalize_output=params['clf_normalize_output'],
            predict_heading=params['clf_predict_heading'],
            sequence_length=params['clf_history'],
        )
    debug_ghost = True
    # overwritting command line params with those the model was trained on
    params = json.load(open(osp.join(folder, 'params.json'), 'r'))
    obs_dict = generate_obs_dict(params)
else:
    classifier = None
    debug_ghost = False

# The classifier will only work for the environment it was trained on,
# so need to set a seed to make sure the same maze is generated
np.random.seed(params['seed'])


# using WASD instead of arrow keys for consistency
UP = 119  # W
LEFT = 97  # A
DOWN = 115  # S
RIGHT = 100  # D

SHUTDOWN = 99  # C

# Define objects to be used as goals
# If the location is set to None they will be set to random accessible locations (each episode?)
object_locations = OrderedDict()
for i in range(args.n_objects):
    object_locations[possible_objects[i]] = None

map_array = generate_maze(map_style=params['map_style'], side_len=params['map_size'])

env = GridWorldEnv(
    map_array=map_array,
    object_locations=object_locations,
    observations=obs_dict,
    movement_type=params['movement_type'],
    max_lin_vel=params['max_lin_vel'],
    max_ang_vel=params['max_ang_vel'],
    continuous=params['continuous'],
    max_steps=params['episode_length'],
    fixed_episode_length=params['fixed_episode_length'],
    dt=params['dt'],
    debug_ghost=debug_ghost,
    classifier=classifier,
    screen_width=300,
    screen_height=300,
)


def keyboard_action(params):
    action_str = getch()
    if params['continuous']:
        # These actions work for both directional and holonomic
        if ord(action_str) == UP:
            action = np.array([1, 0])
        elif ord(action_str) == DOWN:
            action = np.array([-1, 0])
        elif ord(action_str) == LEFT:
            action = np.array([0, -1])
        elif ord(action_str) == RIGHT:
            action = np.array([0, 1])
        else:
            action = np.array([0, 0])
    else:
        # These actions work for both directional and holonomic
        if ord(action_str) == UP:
            action = 0
        elif ord(action_str) == DOWN:
            action = 3  # NOTE: this is not an option for discrete directional
            if params['movement_type'] == 'directional':
                action = 0
        elif ord(action_str) == LEFT:
            action = 1
        elif ord(action_str) == RIGHT:
            action = 2
        else:
            action = 0

    # Set a signal to close the environment
    if ord(action_str) == SHUTDOWN:
        action = None

    return action


num_episodes = 10
time_steps = 10000#100
returns = np.zeros((num_episodes,))
for e in range(num_episodes):
    obs = env.reset(goal_distance=params['goal_distance'])
    for s in range(params['episode_length']):
        env.render()
        env._render_extras()
        #action = env.action_space.sample()
        action = keyboard_action(params)

        # If a specific key is pressed, close the script
        if action is None:
            env.close()
            sys.exit(0)

        obs, reward, done, info = env.step(action)
        # print(obs)
        returns[e] += reward
        # if reward != 0:
        #    print(reward)
        # time.sleep(dt)
        if done:
            break

print(returns)
