import argparse
import json
import numpy as np

from gym_maze.envs.generators import SimpleMazeGenerator, RandomMazeGenerator, \
        RandomBlockMazeGenerator, TMazeGenerator, WaterMazeGenerator


def add_map_arguments(parser):
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--dt', type=float, default=0.1)  # TODO is this the correct default dt?
    # parser.add_argument('--map', type=int, default=0, help='index of the map to use')
    # TODO: have options to train with different maps or the same map
    parser.add_argument('--map-style', type=str, default='blocks', choices=['maze', 'blocks', 't', 'morris', 'simple'],
                        help='maze generator to use')
    parser.add_argument('--map-size', type=int, default=10, help='side length of the maps to generate')
    parser.add_argument('--full-map-obs', action='store_true', help='use the full occupancy map as an observation')
    parser.add_argument('--pob', type=int, default=0, help='partial occupancy observation window')
    parser.add_argument('--n-sensors', type=int, default=5, help='number of distance sensors')
    parser.add_argument('--max-sensor-dist', type=int, default=10, help='maximum distance a sensor can detect')
    parser.add_argument('--normalize-dist-sensors', action='store_true')
    parser.add_argument('--fov', type=float, default=90, help='field of view of distance sensors, in degrees')
    parser.add_argument('--episode-length', type=int, default=1000, help='length of an episode, in timesteps')
    parser.add_argument('--fixed-episode-length', action='store_true',
                        help='task is to get to as many goals as possible in a fixed episode length')
    parser.add_argument('--max-lin-vel', type=float, default=5, help='maximum linear velocity of the agent')
    # parser.add_argument('--min-lin-vel', type=float, default=-1, help='minimum linear velocity of the agent')
    parser.add_argument('--max-ang-vel', type=float, default=5, help='maximum linear angular of the agent')
    parser.add_argument('--fixed-start', action='store_true')
    parser.add_argument('--fixed-goal', action='store_true')
    parser.add_argument('--n-grid-cells', type=int, default=0, help='number of grid cells per scale')
    parser.add_argument('--grid-scales', type=str, default='1', help='comma separated list of grid scales')
    parser.add_argument('--grid-angle', type=float, default=0., help='angle of the grid')
    parser.add_argument('--movement-type', type=str, default='directional',
                        choices=['directional', 'holonomic'],
                        help='type of movement to use')
    parser.add_argument('--heading', type=str, default='angle',
                        choices=['none', 'angle', 'circular', 'normalized_angle', 'map_loc'],
                        help='type of head direction observation to use')
    parser.add_argument('--location', type=str, default='none',
                        choices=['none', 'actual', 'normalized', 'map_loc'],
                        help='type of agent location observation to use')
    parser.add_argument('--goal-loc', type=str, default='none',
                        choices=['none', 'actual', 'normalized', 'map_loc'],
                        help='type of goal observation to use')
    parser.add_argument('--goal-vec', type=str, default='none',
                        choices=['none', 'actual', 'normalized'],
                        help='type of goal displacement observation to use')
    # parser.add_argument('--smell', dest='smell_sensor', action='store_true')
    parser.add_argument('--continuous', action='store_true',
                        help='use continuous actions/observations instead of discrete')
    # parser.add_argument('--step-penalty', type=float, default=0., help='small penalty applied to every step')
    parser.add_argument('--obstacle-ratio', type=float, default=0.2, help='obstacle ratio for generated block mazes')

    # Boundary Cell Parameters - set bc-n-ring to 0 to not use boundary cells
    # currently evenly distributed #TODO: add option for random distribution
    parser.add_argument('--bc-n-ring', type=int, default=0, help='number of boundary cells in the theta dimension')
    parser.add_argument('--bc-n-rad', type=int, default=0, help='number of boundary cells in the radius dimension')
    parser.add_argument('--bc-dist-rad', type=float, default=.75,
                        help='distance between boundary cells in the radius direction')
    parser.add_argument('--bc-receptive-field-min', type=float, default=1,
                        help='receptive field size of the closest cells to the agent')
    parser.add_argument('--bc-receptive-field-max', type=float, default=1.5,
                        help='receptive field size of the furthest cells from the agent')

    # Head Direction Cell Parameters - set bc-n-cells to 0 to not use head direction cells
    # currently evenly distributed #TODO: add option for random distribution
    parser.add_argument('--hd-n-cells', type=int, default=0, help='number of head direction cells')
    parser.add_argument('--hd-receptive-field-min', type=float, default=np.pi / 4,
                        help='smallest receptive field a head direction cell could have')
    parser.add_argument('--hd-receptive-field-max', type=float, default=np.pi / 4,
                        help='largest receptive field a head direction cell could have')

    # Continuous Semantic Pointer Parameters
    # TODO: should both agent and goal use the same x-y axis vectors? Seems like they should
    parser.add_argument('--goal-csp', action='store_true', help='use a semantic pointer for the goal location')
    # parser.add_argument('goal-csp-dim', type=int, default=256, help='dimensionality of the goal CSP')
    # TODO: have options for egocentric with and without head direction taken into account
    parser.add_argument('--goal-csp-egocentric', action='store_true', help='compute the goal CSP relative to the agent')

    parser.add_argument('--agent-csp', action='store_true', help='use a semantic pointer for the agent location')
    # parser.add_argument('agent-csp-dim', type=int, default=256, help='dimensionality of the agent CSP')

    parser.add_argument('--csp-dim', type=int, default=256, help='dimensionality of the continuous semantic pointer')

    parser.add_argument('--param-file', type=str, default='',
                        help='optional parameter file to load from to overwrite command line arguments')

    return parser


def add_ppo_arguments(parser):

    parser.add_argument('--ppo-timesteps', type=int, default=1000000, help='max number of timesteps for ppo to train for')
    # This parameter is typically initialized to 2048, but since the episode length is now longer, it is being increased
    # so that multiple episodes can occur within each batch. Otherwise the batches may be too noisy
    parser.add_argument('--ppo-batchsize', type=int, default=16384, help='number of timesteps per ppo batch')

    return parser


def args_to_dict(args):
    params = vars(args)

    # If a file is supplied to load the parameters from, load it and override the appropriate defaults
    if params['param_file']:
        with open(params['param_file'], "r") as f:
            file_params = json.load(f)

        # Merge dictionaries, replacing elements in 'params' with 'file_params'
        params = {**params, **file_params}

    return params
