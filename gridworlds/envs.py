import numpy as np
from gym import spaces
import sys
from gridworlds import map_utils
# import scipy
import gym
import math
import os
from collections import OrderedDict
from gridworlds import constants
try:
    from gridworlds import csp_utils
except ImportError:
    print("Could not import csp_utils. Make sure Nengo is installed if using CSP observations.")

# Convert a string colour to rgb arguments with: *to_rgb(my_str)
from matplotlib.colors import to_rgb

# if "DISPLAY" in os.environ:
#     from gym.envs.classic_control import rendering
# else:
#     print("No Display detected, skipping rendering imports")

display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    print("No Display detected, skipping rendering imports")
else:
    from gym.envs.classic_control import rendering


def rotate_vector(vec, rot_axis, theta):
    axis = rot_axis.copy()
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.dot(np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]]),
                  vec)


# TODO: add boundary cells and head direction cells,
#       possiblly place cells as well (or learn them)
# TODO: add renderer for boundary cells, head direction cells, grid cells, and place cells
#       most likely have a separate window/viewer for each of these
class GridWorldEnv(gym.Env):

    # Action constants for discrete environment
    FORWARD = constants.FORWARD
    UP = constants.UP
    LEFT = constants.LEFT
    RIGHT = constants.RIGHT
    DOWN = constants.DOWN

    # extra viewers for specific visualizations
    hd_viewer = None  # head direction
    gc_viewer = None  # grid cell
    bc_viewer = None  # boundary cell
    pc_viewer = None  # place cell

    # will be populated with numpy arrays containing the cell activations
    hd_activations = None
    gc_activations = None
    bc_activations = None
    pc_activations = None

    _seed = 0  # for gym compatibility  # TODO: actually use this

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,
                 map_array,
                 object_locations=OrderedDict(),
                 observations=OrderedDict(),
                 movement_type='directional',
                 max_lin_vel=1,
                 max_ang_vel=np.pi/4,
                 continuous=False,
                 max_steps=1000,
                 fixed_episode_length=False,
                 dt=.1,
                 wall_penalty=-1.,
                 movement_cost=-.01,
                 goal_reward=1.,
                 fixed_start=False,
                 fixed_goal=False,
                 debug_ghost=False,
                 classifier=None,
                 screen_width=600,
                 screen_height=600,
                 debug_mode=False,
                 csp_scaling=1,  # multiply state by this value before creating a csp
                 csp_offset=0,  # subtract this value from state before creating a csp
    ):
        """
        GridWorld environment compatible with Gym

        :param map_array: 2D array containing the map information
        :param object_locations: dictionary of object name and location pairs.
                                 If supplied the goal each episode will be to get to a given object.
                                 If no location is specified, a random one will be chosen each reset
        :param movement_type: 'directional' or 'holonomic'
        :param max_lin_vel: maximum linear distance in a single step (continuous control)
        :param max_ang_vel: maximum angular distance in a single step (directional continuous control)
        :param continuous: if True, action and state are continuous, if False, they are discrete
        :param max_steps: maximum number of steps in an episode
        :param fixed_episode_length: each episode is a fixed length, and the task is to get to as many goals as possible
        :param dt: time constant for continuous movement
        :param wall_penalty: reward value for hitting a wall
        :param movement_cost: reward value for moving into free space
        :param goal_reward: reward value for reaching the goal
        :param fixed_start: if true, agent will start at the same location every episode
        :param fixed_goal: if true, the goal will be at the same location every episode
        :param debug_ghost: if true, render an agent where a classifier thinks it is based on current observations
        :param classifier: the classifier function to use for predicting the agent's location
        :param screen_width: width of the rendering windows
        :param screen_height: height of the rendering windows
        :param debug_mode: if true, render/print extra things for debugging
        :param observations: dictionary with keys corresponding to the types of observations available to the agent
                             presence of a key indicates the observation will be returned
                             values of the key indicate values of parameters associated with that observation
                             observations are returned in the order given. Possible options so far are:
                             full_map, pob_view, dist_sensors, heading, location, grid_cells, goal_loc, goal_vec

            full_map: occupancy grid of the full map is given as an observation
            pob_view: partial view of the map is given around the agent
                pob_size: defines the size of the view in terms of number of cells in each direction
            dist_sensors: returns the output of distance sensors attached to the agent
                n_sensors: the number of sensors
                fov_rad: the field of view of the sensor array (in radians)
                max_dist: the maximum distance that can be sensed
                normalize: if true, normalize the output between 0 and 1
            heading: the allocentric orientation of the agent
                TODO: add a version that uses a ring of head direction cells
                circular: if true, output the 2D vector of cos(heading) and sin(heading)
                normalize: if true, normalize the output between -1 and 1
                map_loc: if true and in discrete mode, return a 4D 1-hot vector for the angle.
                         if location is also map_loc, return a (4, width, height) array with a single 1 and the rest 0
            location: the allocentric x-y position of the agent in the map
                normalize: if true, normalize the output between -1 and 1
                map_loc: if true, return an array the shape of the map with a 1 where the agent is located
            grid_cell: activation of simulated grid cells
                n_grid_cells: number of grid cells
                grid_angle: angle of the grid cells wrt a horizontal wall
                grid_scale_range: min and max values of the range of different scales to generate the cells from
                gc_loc (optional): point on a unit hexagon for each grid cell. Will be generated if not specified
                gc_scale (optional): scaling for each grid cell. Will be generated if not specified
            goal_loc: allocentric x-y position of the goal
                normalize: if true, normalize the output between -1 and 1
                map_loc: if true, return an array the shape of the map with a 1 where the goal is located
            goal_vec: distance vector from the agent to the goal
                normalize: if true, normalize the output between -1 and 1
            boundary_cell: activation of simulated boundary cells
                n_ring: number of cells in a ring around the agent
                n_rad: number of cells extending along the radius away from the agent
                dist_rad: distance between the centers of the receptive fields radially
                receptive_field_min: parameter for the smaller receptive field near the agent
                receptive_field_max: parameter for the largest receptive field away from the agent
                TODO: could there be a gaussian std and a cutoff for the receptive fields?
                NOTE: in the real brain these cells will be less organized and more randomly distributed, that should be an option in the future
            goal_csp: continuous semantic pointer representing the goal location
                dim: dimensionality of the semantic pointer
                x_axis_vec: vector for the x_axis
                y_axis_vec: vector for the y_axis
                egocentric: if true, coordinate is relative to the agent
            agent_csp: continuous semantic pointer representing the location of the agent
                dim: dimensionality of the semantic pointer
                x_axis_vec: vector for the x_axis
                y_axis_vec: vector for the y_axis
        """

        assert movement_type in ['directional', 'holonomic']

        self.debug_mode = debug_mode

        self.csp_offset = csp_offset
        self.csp_scaling = csp_scaling

        self.map_array = map_array
        self.map_shape = self.map_array.shape

        self.object_locations = object_locations
        # The name of the object that is the goal for the particular episode
        self.goal_object = ''
        # Store the number of possible goal objects
        if len(self.object_locations) > 0:
            self.n_goal_objects = len(self.object_locations.keys())
            # Fixed list of the string names of the goals
            self.goal_object_list = list(self.object_locations.keys())
            self.goal_object_index = 0
        else:
            self.n_goal_objects = 0
            self.goal_object_list = []
            self.goal_object_index = -1

        self.movement_type = movement_type
        self.continuous = continuous
        self.max_steps = max_steps
        self.fixed_start = fixed_start
        self.fixed_goal = fixed_goal
        self.observations = observations

        self.fixed_episode_length = fixed_episode_length

        self.debug_ghost = debug_ghost
        self.classifier = classifier

        self.width = self.map_array.shape[0]
        self.height = self.map_array.shape[1]

        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.dt = dt

        self.wall_penalty = wall_penalty
        self.movement_cost = movement_cost
        self.goal_reward = goal_reward

        # Dictionary mapping from observation name to the indices of the observation vector where they are found
        # generated inside '_build_observation_space()'
        self.obs_index_dict = {}

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        # x, y, th of the agent
        self.state = np.zeros((3,))

        # Only used if 'fixed_start' is True
        self.start_state = np.zeros((3,))

        # TODO: should th be used for the goal? or just x-y?
        # x, y, th of the goal
        self.goal_state = np.zeros((3,))

        # Set up fixed start and/or goal if needed
        if self.fixed_start:
            # Choose a random starting location
            self.start_state[[0, 1]] = self.random_free_space(continuous=self.continuous)
            # Choose a random starting heading if the movement type is directional
            if self.movement_type == 'directional':
                if self.continuous:
                    self.state[2] = np.random.uniform(low=-np.pi, high=np.pi)
                else:
                    self.state[2] = np.random.randint(low=0, high=4)

        if self.fixed_goal:
            # TODO: more complicated environments could have a more complex goal
            # Choose a random goal location
            self.goal_state[[0, 1]] = self.random_free_space()
            if self.fixed_start:
                # Pick a new goal if the goal is the same as the starting location
                # TODO: in continuous space, this equality check won't be enough
                while (self.start_state[[0, 1]] == self.goal_state[[0, 1]]).all():
                    self.goal_state[[0, 1]] = self.random_free_space()
            # Populate the goal array, in case the observations use this representation
            self.goal_array = np.zeros_like(self.map_array)
            self.goal_array[int(self.goal_state[0]), int(self.goal_state[1])] = 1
        else:
            self.goal_array = np.zeros_like(self.map_array)

        # If any object locations haven't been specified, set them to random locations now
        for name in self.goal_object_list:
            if self.object_locations[name] is None:
                self.object_locations[name] = self.random_free_space(continuous=continuous)

        # keeping track of steps taken in the episode
        self.step_count = 0

        # State transition mappings for discrete environments
        # given movement_action, compute displacement
        self.holonomic_transitions = constants.holonomic_transitions

        # given (direction_action, current_heading) produce (next_heading)
        self.directional_transitions = constants.directional_transitions

        # keep track of the observations, so they can be accessed internally by the renderer
        self.obs = self._get_obs()

        ##########################
        # Variables for renderer #
        ##########################
        # NOTE: assuming square environment
        if screen_width != screen_height:
            raise NotImplementedError("Environment currently assumes screen_width = screen_height")
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.scale = self.screen_width/self.width

        # The width and height of each square tile
        self.tile_size = self.scale

        self.viewer = None

    def _build_action_space(self):

        if self.continuous:
            if self.movement_type == 'directional':
                # linear, angular
                return gym.spaces.Box(
                    low=-1*np.ones((2,)),
                    high=np.ones((2,)),
                )
            elif self.movement_type == 'holonomic':
                # x, y
                return gym.spaces.Box(
                    low=-1*np.ones((2,)),
                    high=np.ones((2,)),
                )
            else:
                raise NotImplementedError(
                    'movement type of {0} not implemented for continuous control'.format(self.movement_type)
                )
        else:
            if self.movement_type == 'directional':
                # forward, left, right
                return gym.spaces.Discrete(3)
            elif self.movement_type == 'holonomic':
                # up, down, left, right
                return gym.spaces.Discrete(4)
            else:
                raise NotImplementedError(
                    'movement type of {0} not implemented for discete control'.format(self.movement_type)
                )

    def _build_observation_space(self):

        self.obs_low = list()
        self.obs_high = list()

        # full_map, pob_view, dist_sensors, heading, location, grid_cells, goal_loc, goal_vec
        for obs in self.observations:
            index_start = len(self.obs_low)
            if obs == 'full_map':
                self.obs_low += list(np.zeros_like(self.map_array).flatten())
                self.obs_high += list(np.ones_like(self.map_array).flatten())
            elif obs == 'pob_view':
                pob_size = self.observations[obs]['pob_size']
                self.obs_low += list(np.zeros((pob_size, pob_size)).flatten())
                self.obs_high += list(np.ones((pob_size, pob_size)).flatten())
            elif obs == 'dist_sensors':
                n_sensors = self.observations[obs]['n_sensors']
                normalize = self.observations[obs]['normalize']
                if normalize:
                    max_dist = 1
                else:
                    max_dist = self.observations[obs]['max_dist']
                self.obs_low += [0]*n_sensors
                self.obs_high += [max_dist]*n_sensors
            elif obs == 'heading':
                normalize = self.observations[obs]['normalize']
                circular = self.observations[obs]['circular']
                map_loc = self.observations[obs]['map_loc']
                if map_loc:
                    # Handle the potential interaction with 'location'
                    if 'location' in self.observations and self.observations['location']['map_loc']:
                        # The observations will be handled in the 'location' block, nothing to do here
                        pass
                    else:
                        # 4D one-hot vector for heading
                        self.obs_low += [0] * 4
                        self.obs_high += [1] * 4
                elif circular:
                    self.obs_low += [-1]*2
                    self.obs_high += [1]*2
                elif normalize:
                    self.obs_low += [-1]
                    self.obs_high += [1]
                else:
                    self.obs_low += [-np.pi]
                    self.obs_high += [np.pi]
            elif obs == 'location':
                normalize = self.observations[obs]['normalize']
                map_loc = self.observations[obs]['map_loc']
                if map_loc:
                    # Handle the potential interaction with 'heading'
                    if 'heading' in self.observations and self.observations['heading']['map_loc']:
                        self.obs_low += list(np.zeros(4 * self.map_array.shape[0] * self.map_array.shape[1]))
                        self.obs_high += list(np.ones(4 * self.map_array.shape[0] * self.map_array.shape[1]))
                    else:
                        self.obs_low += list(np.zeros_like(self.map_array).flatten())
                        self.obs_high += list(np.ones_like(self.map_array).flatten())
                elif normalize:
                    self.obs_low += [-1]*2
                    self.obs_high += [1]*2
                else:
                    self.obs_low += [0]*2
                    self.obs_high += [max(self.width, self.height)]*2
            elif obs == 'grid_cells':
                n_grid_cells = self.observations[obs]['n_grid_cells']
                self.obs_low += [0]*n_grid_cells
                self.obs_high += [1]*n_grid_cells
            elif obs == 'goal_loc':
                normalize = self.observations[obs]['normalize']
                map_loc = self.observations[obs]['map_loc']
                if map_loc:
                    self.obs_low += list(np.zeros_like(self.map_array).flatten())
                    self.obs_high += list(np.ones_like(self.map_array).flatten())
                elif normalize:
                    self.obs_low += [-1]*2
                    self.obs_high += [1]*2
                else:
                    self.obs_low += [0]*2
                    self.obs_high += [max(self.width, self.height)]*2
            elif obs == 'goal_vec':
                normalize = self.observations[obs]['normalize']
                if normalize:
                    self.obs_low += [-1]*2
                    self.obs_high += [1]*2
                else:
                    self.obs_low += [-max(self.width, self.height)]*2
                    self.obs_high += [max(self.width, self.height)]*2
            elif obs == 'boundary_cell':
                n_ring = self.observations[obs]['n_ring']
                n_rad = self.observations[obs]['n_rad']
                n_cells = n_ring * n_rad
                self.obs_low += [0]*n_cells
                self.obs_high += [1]*n_cells

                # Initialize array to store the activations
                self.bc_activations = np.zeros((n_rad, n_ring))
            elif obs == 'hd_cell':
                n_cells = self.observations[obs]['n_cells']
                self.obs_low += [0] * n_cells
                self.obs_high += [1] * n_cells

                # Initialize array to store the activations
                self.hd_activations = np.zeros((n_cells,))
            elif obs == 'goal_csp':
                dim = self.observations[obs]['dim']
                self.obs_low += [-1] * dim
                self.obs_high += [1] * dim
                #
                # # FIXME: these might not need to be saved, could just get from observations dict
                # self.goal_x_axis_vec = self.observations[obs]['x_axis_vec']
                # self.goal_y_axis_vec = self.observations[obs]['y_axis_vec']
            elif obs == 'agent_csp':
                dim = self.observations[obs]['dim']
                self.obs_low += [-1] * dim
                self.obs_high += [1] * dim
                #
                # # FIXME: these might not need to be saved, could just get from observations dict
                # self.agent_x_axis_vec = self.observations[obs]['x_axis_vec']
                # self.agent_y_axis_vec = self.observations[obs]['y_axis_vec']
            else:
                raise NotImplementedError("Unrecognized observation type {0}".format(obs))
            index_end = len(self.obs_low)
            self.obs_index_dict[obs] = list(range(index_start, index_end))

        return gym.spaces.Box(
            low=np.array(self.obs_low),
            high=np.array(self.obs_high),
        )

    def _get_obs(self, return_dict=False):
        """
        Create the observation vector
        if 'return_dict' is true, return as a dictionary instead
        """

        obs_values = []
        obs_dict = {}

        # full_map, pob_view, dist_sensors, heading, location, grid_cells, goal_loc, goal_vec
        for obs in self.observations:
            if obs == 'full_map':
                if return_dict:
                    obs_dict[obs] = self.map_array
                else:
                    obs_values += list(self.map_array.flatten())
            elif obs == 'pob_view':
                pob_size = self.observations[obs]['pob_size']
                if return_dict:
                    obs_dict[obs] = self.get_pob_view(pob_size=pob_size)
                else:
                    obs_values += list(self.get_pob_view(pob_size=pob_size).flatten())
            elif obs == 'dist_sensors':
                # TODO FIXME
                n_sensors = self.observations[obs]['n_sensors']
                fov_rad = self.observations[obs]['fov_rad']
                max_dist = self.observations[obs]['max_dist']
                normalize = self.observations[obs]['normalize']
                sensors = self.get_dist_sensor_readings(
                    state=self.state,
                    n_sensors=n_sensors,
                    fov_rad=fov_rad,
                    max_dist=max_dist,
                    normalize=normalize,
                )
                if return_dict:
                    obs_dict[obs] = sensors
                else:
                    obs_values += list(sensors)
            elif obs == 'heading':
                normalize = self.observations[obs]['normalize']
                circular = self.observations[obs]['circular']
                map_loc = self.observations[obs]['map_loc']
                th = self.state[2]
                if not self.continuous:
                    # convert direction integers to angle in radians, to be consistent
                    th = self._dir_to_ang(self.state[2])
                if map_loc:
                    # Handle the potential interaction with 'location'
                    if 'location' in self.observations and self.observations['location']['map_loc']:
                        # The observations will be handled in the 'location' block, nothing to do here
                        pass
                    else:
                        # 4D one-hot vector for heading
                        ang_vec = np.zeros((4,))
                        ang_vec[self.angle_to_index(self.state[2])] = 1  # TODO: is a simple integer conversion correct here in all cases? need to add to the correct quadrant from angle
                        if return_dict:
                            obs_dict[obs] = ang_vec
                        else:
                            obs_values += list(ang_vec)
                elif circular:
                    if return_dict:
                        obs_dict[obs] = np.array([np.cos(th), np.sin(th)])
                    else:
                        obs_values += [np.cos(th), np.sin(th)]
                elif normalize:
                    if return_dict:
                        obs_dict[obs] = th/np.pi
                    else:
                        obs_values += [th/np.pi]
                else:
                    if return_dict:
                        obs_dict[obs] = th
                    else:
                        obs_values += [th]
            elif obs == 'location':
                normalize = self.observations[obs]['normalize']
                map_loc = self.observations[obs]['map_loc']
                if map_loc:
                    # Handle the potential interaction with 'heading'
                    # TODO: is int() the correct way to map continuous to discrete?

                    if 'heading' in self.observations and self.observations['heading']['map_loc']:
                        # NOTE: this is assuming discrete angles represented as 0-3
                        if self.continuous:
                            ang_index = self.angle_to_index(self.state[2])
                        else:
                            ang_index = self.state[2]  # in the discrete case, angle is already an integer
                        arr = np.zeros((4, self.width, self.height))
                        x = np.clip(int(np.round(self.state[0])), 0, self.width - 1)
                        y = np.clip(int(np.round(self.state[1])), 0, self.height - 1)
                        arr[ang_index, x, y] = 1
                    else:
                        arr = np.zeros((self.width, self.height))
                        arr[int(self.state[0]), int(self.state[1])] = 1

                    if return_dict:
                        obs_dict[obs] = arr
                    else:
                        obs_values += list(arr.flatten())
                elif normalize:
                    x = (self.state[0] / self.width) * 2 - 1
                    y = (self.state[1] / self.height) * 2 - 1
                    if return_dict:
                        obs_dict[obs] = np.array([x, y])
                    else:
                        obs_values += [x, y]
                else:
                    if return_dict:
                        obs_dict[obs] = self.state[[0, 1]].copy()
                    else:
                        obs_values += list(self.state[[0, 1]])
            elif obs == 'grid_cells':
                n_grid_cells = self.observations[obs]['n_grid_cells']
                # TODO:
            elif obs == 'goal_loc':
                normalize = self.observations[obs]['normalize']
                map_loc = self.observations[obs]['map_loc']
                if map_loc:
                    if return_dict:
                        obs_dict[obs] = self.goal_array.copy()
                    else:
                        obs_values += list(self.goal_array.flatten())
                elif normalize:
                    x = (self.goal_state[0] / self.width) * 2 - 1
                    y = (self.goal_state[1] / self.height) * 2 - 1
                    if return_dict:
                        obs_dict[obs] = np.array([x, y])
                    else:
                        obs_values += [x, y]
                else:
                    if return_dict:
                        obs_dict[obs] = self.goal_state[[0, 1]].copy()
                    else:
                        obs_values += list(self.goal_state[[0, 1]])
            elif obs == 'goal_vec':
                normalize = self.observations[obs]['normalize']
                if normalize:
                    x = ((self.goal_state[0] - self.state[0]) / self.width) * 2 - 1
                    y = ((self.goal_state[1] - self.state[1]) / self.height) * 2 - 1
                    if return_dict:
                        obs_dict[obs] = np.array([x, y])
                    else:
                        obs_values += [x, y]
                else:
                    if return_dict:
                        obs_dict[obs] = self.goal_state[[0, 1]] - self.state[[0, 1]]
                    else:
                        obs_values += list(self.goal_state[[0, 1]] - self.state[[0, 1]])
            elif obs == 'boundary_cell':
                # this function will populate bc_activations with the correct values
                # TODO: it needs to know stuff like dist_rad and the mix/max receptive field
                self.compute_boundary_cell_activations()
                if return_dict:
                    obs_dict[obs] = self.bc_activations.copy()
                else:
                    obs_values += list(self.bc_activations.flatten())
            elif obs == 'hd_cell':
                # this function will populate hd_activations with the correct values
                # TODO: it needs to know stuff like n_cells and the mix/max receptive field
                self.compute_head_direction_cell_activations()
                if return_dict:
                    obs_dict[obs] = self.hd_activations.copy()
                else:
                    obs_values += list(self.hd_activations.flatten())
            elif obs == 'goal_csp':
                # TODO: add option for normalization here in the future?
                if self.observations[obs]['egocentric']:
                    x = self.goal_state[0] - self.state[0]
                    y = self.goal_state[1] - self.state[1]
                else:
                    x = self.goal_state[0]
                    y = self.goal_state[1]
                vec = csp_utils.encode_point(
                    x=(x - self.csp_offset) * self.csp_scaling,
                    y=(y - self.csp_offset) * self.csp_scaling,
                    x_axis_vec=self.observations[obs]['x_axis_vec'],
                    y_axis_vec=self.observations[obs]['y_axis_vec']
                ).v
                if return_dict:
                    obs_dict[obs] = vec
                else:
                    obs_values += list(vec)

            elif obs == 'agent_csp':
                # TODO: add option for normalization here in the future?
                x = self.state[0]
                y = self.state[1]
                vec = csp_utils.encode_point(
                    x=(x - self.csp_offset) * self.csp_scaling,
                    y=(y - self.csp_offset) * self.csp_scaling,
                    x_axis_vec=self.observations[obs]['x_axis_vec'],
                    y_axis_vec=self.observations[obs]['y_axis_vec']
                ).v
                if return_dict:
                    obs_dict[obs] = vec
                else:
                    obs_values += list(vec)

        if return_dict:
            return obs_dict
        else:
            return np.array(obs_values)

    # def _get_reward(self):
    #
    #     pass

    # def _check_done(self):
    #
    #     pass

    def angle_to_index(self, ang):
        """
        converts a given angle (in radians) to an index corresponding to the closest cardinal direction
        used for a particular observation type (continuous directional control with map_loc heading observations)
        """
        # NOTE: this is assuming angles between -pi and pi
        if ang < -np.pi*3/4 or ang > np.pi*3/4:
            return 3
        elif ang < -np.pi/4:
            return 2
        elif ang < np.pi/4:
            return 1
        elif ang <= np.pi*3/4:
            return 0

    def _get_reward_done(self, new_state, old_state):

        # TODO: implement options for a more sophisticated reward function

        if (new_state == old_state).all():
            # large penalty for hitting a wall and not moving
            reward = self.wall_penalty  # -1
        else:
            # Small negative movement penalty
            reward = self.movement_cost  # -0.01

        done = self._goal_check()

        if done:
            reward = self.goal_reward  # 1

        return reward, done

    def _goal_check(self):

        if np.linalg.norm(self.state[[0, 1]] - self.goal_state[[0, 1]]) < 1:
            return True
        else:
            return False

    #def _step(self, action):
    def step(self, action):

        old_state = self.state.copy()

        self._update_state(action)

        reward, done = self._get_reward_done(self.state, old_state)

        # If running in fixed episode mode, teleport the agent to a new start location
        # but still give them the reward for getting to the goal
        if done and self.fixed_episode_length:
            done = False
            self._reset_agent()

        self.obs = self._get_obs()

        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True

        # done = self._check_done()

        info = {}

        return self.obs, reward, done, info

    def _update_state(self, action):

        # amount to move by
        displacement = np.zeros((2,))
        if self.continuous:
            #TODO: need appropriate collision detection here:
            #      should there be a negative reward if the wall is hit, or is slower movement enough of a penalty?
            if self.movement_type == 'holonomic':
                # clip action within maximum values
                lin_vel = np.clip(action[[0, 1]], -self.max_lin_vel, self.max_lin_vel)
                displacement = lin_vel * self.dt
            elif self.movement_type == 'directional':
                ang_vel = np.clip(action[1], -self.max_ang_vel, self.max_ang_vel)
                lin_vel = np.clip(action[0], -self.max_lin_vel, self.max_lin_vel)
                self.state[2] += ang_vel * self.dt
                if self.state[2] > np.pi:
                    self.state[2] -= 2*np.pi
                elif self.state[2] < -np.pi:
                    self.state[2] += 2*np.pi
                displacement = np.array([np.cos(self.state[2]), np.sin(self.state[2])])*lin_vel*self.dt
            else:
                raise NotImplementedError(
                    'movement type of {0} not implemented for continuous control'.format(self.movement_type)
                )
            new_pos = self.state[[0, 1]] + displacement
            if self.free_space(new_pos):
                self.state[[0, 1]] = new_pos
            else:
                # TODO: move as close as possible to the wall
                return
        else:
            if self.movement_type == 'directional':

                if action == self.FORWARD:
                    # move forward in the direction currently facing
                    displacement = self.holonomic_transitions[self.state[2]]
                else:
                    # rotation
                    self.state[2] = self.directional_transitions[(action, self.state[2])]
                    return
            elif self.movement_type == 'holonomic':
                displacement = self.holonomic_transitions[action]
            else:
                raise NotImplementedError(
                    'movement type of {0} not implemented for discrete control'.format(self.movement_type)
                )

            new_pos = (self.state[[0, 1]] + displacement).astype(int)
            new_pos = np.clip(new_pos, [0, 0], [self.width - 1, self.height - 1])
            if self.map_array[new_pos[0], new_pos[1]] == 0:
                self.state[[0, 1]] = new_pos
            else:
                return

    def _reset_agent(self):
        # Choose a random starting location
        self.state[[0, 1]] = self.random_free_space(continuous=self.continuous)

        # Choose a random starting heading if the movement type is directional
        if self.movement_type == 'directional':
            if self.continuous:
                self.state[2] = np.random.uniform(low=-np.pi, high=np.pi)
            else:
                self.state[2] = np.random.randint(low=0, high=4)

    def _reset_goal(self, goal_distance=0):
        """
        Choose a random goal location. If a list of goal objects was given to this environment,
        then choose at random from those objects' locations
        :param goal_distance: optional parameter for limiting distance of non-object goals from the start location
        """

        # TODO: more complicated environments could have a more complex goal

        if self.n_goal_objects == 0:
            # Choose a random goal location
            if goal_distance > 0:
                self.goal_state[[0, 1]] = self.constrained_free_space(center=self.state[[0, 1]], distance=goal_distance)
            else:
                self.goal_state[[0, 1]] = self.random_free_space()

            # Pick a new goal if the goal is the same as the starting location
            # TODO: in continuous space, this equality check won't be enough
            while (self.state[[0, 1]] == self.goal_state[[0, 1]]).all():
                if goal_distance > 0:
                    self.goal_state[[0, 1]] = self.constrained_free_space(center=self.state[[0, 1]],
                                                                          distance=goal_distance)
                else:
                    self.goal_state[[0, 1]] = self.random_free_space()
        else:
            # Choose a random object to make the goal
            # TODO: there are a lot of redundant variables here, clean this up in the future
            self.goal_object_index = int(np.random.randint(low=0, high=self.n_goal_objects, size=1))
            self.goal_object = self.goal_object_list[self.goal_object_index]
            self.goal_state[[0, 1]] = self.object_locations[self.goal_object]

        # Populate the goal array, in case the observations use this representation
        self.goal_array = np.zeros_like(self.map_array)
        self.goal_array[int(self.goal_state[0]), int(self.goal_state[1])] = 1

    def reset(self, goal_distance=0):
        """
        Reset the environment, with an optional goal_distance parameter to force the chosen goal to be
        within a certain distance of the start location. Useful when training a network in a curriculum learning manner.
        If the goal_distance is set to 0, any distance can be used
        """

        # TODO: when using goal_distance, it should really be shortest path length rather than bird's eye distance
        #       to the goal, because walls could be in the way and actually make it a lot longer. Hopefully these cases
        #       are rare enough that a curriculum learning setting will still work fine
        #       In continuous cases might want to take initial heading into account as well. The goal being
        #       close but in the opposite direction may be harder than a slightly further goal in the same direction

        # TODO: add the possibility to change the map on reset (choose from a list of given maps, or generate one)

        if self.fixed_start:
            self.state = self.start_state.copy()
        else:
            self._reset_agent()

        if not self.fixed_goal:
            self._reset_goal(goal_distance=goal_distance)

        self.step_count = 0

        if self.viewer is not None:
            # Create new viewer with the updated goal location
            self.goal_trans.set_translation(
                self._scale_x_pos(self.goal_state[0]),
                self._scale_y_pos(self.goal_state[1]),
            )

            # modify goal object images if needed here
            i = 0
            for name, loc in self.object_locations.items():
                self.obj_img_trans[i].set_translation(
                    self._scale_x_pos(loc[0]),
                    self._scale_y_pos(loc[1]),
                )
                i += 1

        self.obs = self._get_obs()

        return self.obs

    def set_agent_state(self, state):
        """
        Debug function that teleports the agent to a particular state (x,y,th)
        """
        self.state = state

    def set_goal_state(self, state):
        """
        Debug function that sets the goal to a particular state (x,y,th)
        """
        self.goal_state = state

        # Populate the goal array, in case the observations use this representation
        self.goal_array = np.zeros_like(self.map_array)
        self.goal_array[int(self.goal_state[0]), int(self.goal_state[1])] = 1

    def free_space(self, pos):
        """
        Returns True if the position corresponds to a free space in the map
        :param pos: 2D floating point x-y coordinates
        """
        # TODO: doublecheck that rounding is the correct thing to do here
        x = np.clip(int(np.round(pos[0])), 0, self.width - 1)
        y = np.clip(int(np.round(pos[1])), 0, self.height - 1)
        return self.map_array[x, y] == 0

    def get_pob_view(self, pob_size=1):
        """
        Return the partial observation view around the current state
        Assume anything out of bounds is an obstacle (use a padded environment to make this faster?)
        TODO: should interpolation happen for continuous environments?
        """
        ret = np.ones((pob_size, pob_size))
        raise NotImplementedError

        return ret

    def get_dist_sensor_readings(self, state, n_sensors, fov_rad, max_dist, normalize):
        """
        Returns distance sensor readings from a given state
        """

        sensors = map_utils.generate_sensor_readings(
            map_arr=self.map_array,
            # zoom_level=20,#20,#4,
            n_sensors=n_sensors,
            fov_rad=fov_rad,
            x=state[0],  # this way works best for the fixed sensors
            y=state[1],
            # x=state[0]+.5,  # seems like it's necessary to add .5 to get correct looking measurements
            # y=state[1]+.5,
            th=state[2],
            max_sensor_dist=max_dist,
            debug_value=0,
        )

        if normalize:
            sensors /= max_dist

        return sensors

    def random_free_space(self, continuous=False):
        """
        Returns the coordinate of a random unoccupied space in the map
        If continuous is True, returns float coordinates rather than integer
        """

        # TODO: make this more efficient
        if continuous:
            while True:
                x, y = np.random.uniform(self.width), np.random.uniform(self.height)
                if self.free_space([x, y]):
                    return np.array([x, y])
        else:
            while True:
                x, y = np.random.randint(self.width), np.random.randint(self.height)
                if self.map_array[x, y] == 0:
                    return np.array([x, y])

    def constrained_free_space(self, center, distance, continuous=False):
        """
        Returns the coordinate of a random unoccupied space in the map
        Constrained to be within a particular distance of a center point
        If continuous is True, returns float coordinates rather than integer
        """

        min_x = max(center[0] - distance, 0)
        max_x = min(center[0] + distance, self.width - 1)

        min_y = max(center[1] - distance, 0)
        max_y = min(center[1] + distance, self.height - 1)

        if continuous:
            while True:
                x, y = np.random.uniform(low=min_x, high=max_x), np.random.uniform(low=min_y, high=max_y)
                if self.free_space([x, y]):
                    return np.array([x, y])
        else:
            while True:
                x, y = np.random.randint(low=min_x, high=max_x), np.random.randint(low=min_y, high=max_y)
                if self.map_array[x, y] == 0:
                    return np.array([x, y])

    def _create_rendering_viewer(self):
        self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

        # Common points for convenience
        tr = (self.tile_size / 2, self.tile_size / 2)
        tl = (self.tile_size / 2, -self.tile_size / 2)
        br = (-self.tile_size / 2, self.tile_size / 2)
        bl = (-self.tile_size / 2, -self.tile_size / 2)

        # Note: order that the polygons are created is the z-order, with the
        # first being on the bottom.

        goal_poly = rendering.FilledPolygon([bl, tl, tr, br])
        goal_poly.set_color(*to_rgb('green'))
        self.goal_trans = rendering.Transform(
            translation=(self._scale_x_pos(self.goal_state[0]),
                         self._scale_y_pos(self.goal_state[1]))
        )
        goal_poly.add_attr(self.goal_trans)
        self.viewer.add_geom(goal_poly)

        agent_poly = rendering.FilledPolygon(
            [
                (-self.tile_size / 2., -self.tile_size / 3.),
                (self.tile_size / 2., 0),
                (-self.tile_size / 2., self.tile_size / 3.),
            ]
        )
        agent_poly.set_color(*to_rgb('blue'))
        # Note: tile_size/2.0 is added because the coordinate is at the center of the tile
        self.agent_trans = rendering.Transform(
            translation=(self._scale_x_pos(self.state[0]),
                         self._scale_y_pos(self.state[1]))
        )

        agent_poly.add_attr(self.agent_trans)
        self.viewer.add_geom(agent_poly)

        if self.debug_ghost:
            ghost_poly = rendering.FilledPolygon(
                [
                    (-self.tile_size / 2., -self.tile_size / 3.),
                    (self.tile_size / 2., 0),
                    (-self.tile_size / 2., self.tile_size / 3.),
                ]
            )
            ghost_poly.set_color(*to_rgb('orange'))
            self.ghost_trans = rendering.Transform(
                translation=(self._scale_x_pos(self.state[0]),
                             self._scale_y_pos(self.state[1]))
            )
            ghost_poly.add_attr(self.ghost_trans)
            self.viewer.add_geom(ghost_poly)

        for x in range(self.width):
            for y in range(self.height):
                if self.map_array[x, y] == 1:
                    wall_poly = rendering.FilledPolygon([bl, tl, tr, br])
                    wall_poly.set_color(*to_rgb('black'))
                    wall_trans = rendering.Transform(
                        translation=(self._scale_x_pos(x),
                                     self._scale_y_pos(y))
                    )
                    wall_poly.add_attr(wall_trans)
                    self.viewer.add_geom(wall_poly)

        # Add goal objects as images if required
        # NOTE: this is assuming the names of the items exist in the assets folder
        # FIXME: allow the images to be loaded from anywhere
        if len(self.object_locations) > 0:
            self.obj_img_trans = []
            for name, loc in self.object_locations.items():
                fname = os.path.join(
                    os.path.dirname(__file__),
                    'assets/icons8-{0}-64.png'.format(name)
                )
                img = rendering.Image(fname=fname, width=self.tile_size, height=self.tile_size)
                self.obj_img_trans.append(
                    rendering.Transform(
                        translation=(self._scale_x_pos(loc[0]),
                                     self._scale_y_pos(loc[1]))
                    )
                )
                img.add_attr(self.obj_img_trans[-1])

                self.viewer.add_geom(img)

        # Lines for the distance sensors
        if 'dist_sensors' in self.observations.keys():
            self.dist_sensor_lines = []
            n_sensors = self.observations['dist_sensors']['n_sensors']
            for i in range(n_sensors):
                line = rendering.Line()
                line.set_color(*to_rgb('red'))
                self.dist_sensor_lines.append(line)
                self.viewer.add_geom(line)

        # Debugging
        if self.debug_mode:
            for x in range(self.map_array.shape[0]):
                x_scale = self._scale_x_pos(x)
                line = rendering.Line(start=(x_scale, 0), end=(x_scale, self.screen_height))
                line.set_color(*to_rgb('green'))
                self.viewer.add_geom(line)
            for y in range(self.map_array.shape[1]):
                y_scale = self._scale_x_pos(y)
                line = rendering.Line(start=(0, y_scale), end=(self.screen_width, y_scale))
                line.set_color(*to_rgb('green'))
                self.viewer.add_geom(line)

            self.crosshair_x = rendering.Line()
            self.crosshair_x.set_color(*to_rgb('purple'))
            self.viewer.add_geom(self.crosshair_x)

            self.crosshair_y = rendering.Line()
            self.crosshair_y.set_color(*to_rgb('purple'))
            self.viewer.add_geom(self.crosshair_y)

            # self.central_lines = []
            # for i in range(self.map_array.shape[0]):
            #     for j in range(self.map_array.shape[1]):
            #         self.central_lines.append(rendering.Line(start=(self._scale_x_pos(i), self._scale_y_pos(j)), end=(0, 0)))
            #         self.central_lines[-1].set_color(*to_rgb('yellow'))
            #         self.viewer.add_geom(self.central_lines[-1])

    def _render(self, mode='human', close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self._create_rendering_viewer()

        agent_x = self._scale_x_pos(self.state[0])
        agent_y = self._scale_y_pos(self.state[1])
        self.agent_trans.set_translation(agent_x, agent_y)
        self.agent_trans.set_rotation(self._scale_theta(self.state[2]))

        if self.debug_ghost and self.classifier is not None:

            obs_dict = self._get_obs(return_dict=True)
            # obs = obs.reshape((1, len(obs)))

            pos = self.classifier(obs_dict)

            ghost_x = self._scale_x_pos(pos[0])
            ghost_y = self._scale_y_pos(pos[1])

            # use heading prediction if given
            if len(pos) == 3:
                ghost_th = pos[2]
            else:
                ghost_th = 0

            self.ghost_trans.set_translation(ghost_x, ghost_y)
            self.ghost_trans.set_rotation(self._scale_theta(ghost_th))

        # Lines for the distance sensors
        if 'dist_sensors' in self.observations.keys():
            n_sensors = self.observations['dist_sensors']['n_sensors']
            fov_rad = self.observations['dist_sensors']['fov_rad']
            normalize = self.observations['dist_sensors']['normalize']
            max_dist = self.observations['dist_sensors']['max_dist']
            dists = self.obs[self.obs_index_dict['dist_sensors']]
            if normalize:
                dists *= max_dist
            ang_interval = np.pi / len(dists)
            start_ang = -fov_rad / 2. + self.state[2]
            angs = np.linspace(-fov_rad / 2. + self.state[2], fov_rad / 2. + self.state[2], n_sensors)
            for i, line in enumerate(self.dist_sensor_lines):
                # x = self._scale_x_pos(self.state[0] + dists[i] * np.cos(start_ang + i * ang_interval))
                # y = self._scale_y_pos(self.state[1] + dists[i] * np.sin(start_ang + i * ang_interval))
                x = self._scale_x_pos(self.state[0] + dists[i] * np.cos(angs[i]))
                y = self._scale_y_pos(self.state[1] + dists[i] * np.sin(angs[i]))
                line.start = (agent_x, agent_y)
                line.end = (x, y)

                # screen_dist = np.sqrt((agent_x - x)**2 + (agent_y - y)**2)
                # scaled_dist = screen_dist / self.scale
                # print(dists[i], scaled_dist)

                # line.end = (x - self.tile_size / 2., y - self.tile_size / 2.)

        # Debugging
        if self.debug_mode:
            self.crosshair_x.start = (agent_x, 0)
            self.crosshair_x.end = (agent_x, self.screen_height)

            self.crosshair_y.start = (0, agent_y)
            self.crosshair_y.end = (self.screen_width, agent_y)

            # for line in self.central_lines:
            #     line.end = (agent_x, agent_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def compute_head_direction_cell_activations(self):
        n_hd_cells = self.observations['hd_cell']['n_cells']
        # TODO: use these receptive field values
        receptive_field_min = self.observations['hd_cell']['receptive_field_min']
        receptive_field_max = self.observations['hd_cell']['receptive_field_max']

        if self.continuous:
            th = self.state[2]
        else:
            th = self._dir_to_ang(self.state[2])

        for i in range(n_hd_cells):
            theta = i * 2*np.pi / n_hd_cells
            # TODO: deal with the wraparound correctly
            # diff = abs(self.state[2]+np.pi - theta)
            diff = min(abs(th + np.pi - theta),
                       abs(th + np.pi - theta + 2*np.pi),
                       abs(th + np.pi - theta - 2*np.pi),
                       )

            if diff < 2*np.pi/n_hd_cells:
                self.hd_activations[i] = 1 - (diff / (2*np.pi/n_hd_cells))
            else:
                self.hd_activations[i] = 0

    def _create_head_direction_viewer(self):
        """
        Viewer for head direction cell activation
        """
        # TODO: should height and width be different than environment?
        self.hd_viewer = rendering.Viewer(self.screen_width, self.screen_height)

        # TODO: fill out this function correctly, for now just something basic
        #       to make sure the screen is working

        # create a set of circles representing each of the hd cells in a ring
        n_hd_cells = self.observations['hd_cell']['n_cells']

        # only need to save the polygons for colour changes
        # don't need to save the transformations since they are not moving
        self.hd_polys = []

        for i in range(n_hd_cells):

            #FIXME: NOTE: adding an extra 90 degrees to align with the viewer
            #             currently looking to the right is 0 degrees in the code
            theta = -i * 2*np.pi / n_hd_cells - np.pi/2.
            circle_poly = rendering.make_circle(radius=self.scale, res=30, filled=True)
            circle_poly.set_color(*to_rgb('black'))
            circle_trans = rendering.Transform(
                translation=(self._scale_x_pos(self.width/2. + 3*np.sin(theta)),
                             self._scale_y_pos(self.height/2. + 3*np.cos(theta)))
            )
            circle_poly.add_attr(circle_trans)
            self.hd_viewer.add_geom(circle_poly)
            self.hd_polys.append(circle_poly)

    def _render_head_direction(self):

        if self.hd_viewer is None:
            self._create_head_direction_viewer()

        for i in range(len(self.hd_activations)):
            self.hd_polys[i].set_color(0, 0, self.hd_activations[i])
            
        return self.hd_viewer.render()
    
    def compute_boundary_cell_activations(self):
        n_bc_ring = self.observations['boundary_cell']['n_ring']
        n_bc_rad = self.observations['boundary_cell']['n_rad']
        dist_rad = self.observations['boundary_cell']['dist_rad']
        receptive_field_min = self.observations['boundary_cell']['receptive_field_min']
        receptive_field_max = self.observations['boundary_cell']['receptive_field_max']
        
        for r in range(n_bc_rad):
            for th in range(n_bc_ring):
                # Center of the receptive field in r-th coordinates
                theta = th * 2*np.pi / n_bc_ring
                rad = (r+1) * dist_rad

                # Center of the receptive field in x-y coordinates
                # offset by the agent's current position to be in world coordinates
                x = self.state[0] + rad * np.sin(theta)
                y = self.state[1] + rad * np.cos(theta)

                # TODO: have a gradient of activation
                if self.free_space((x, y)):
                    self.bc_activations[r, th] = 0
                else:
                    self.bc_activations[r, th] = 1

    def _create_boundary_cell_viewer(self):
        """
        Viewer for boundary cell activation
        """
        # TODO: should height and width be different than environment?
        self.bc_viewer = rendering.Viewer(self.screen_width, self.screen_height)

        # number of boundary cells in a ring around the agent
        n_bc_ring = self.observations['boundary_cell']['n_ring']#12

        # depth of boundary cells from the agent
        # TODO: make further boundary cells wider?
        n_bc_rad = self.observations['boundary_cell']['n_rad']#3
        
        dist_rad = self.observations['boundary_cell']['dist_rad']#3

        self.bc_polys = []
        
        for r in range(n_bc_rad):
            ring_list = []
            for th in range(n_bc_ring):

                rad = (r+1) * dist_rad#1.5
                theta = th * 2*np.pi / n_bc_ring
                circle_poly = rendering.make_circle(radius=self.scale/2.+r, res=30, filled=True)
                circle_poly.set_color(*to_rgb('black'))
                circle_trans = rendering.Transform(
                    translation=(self._scale_x_pos(self.width/2. + rad*np.sin(theta)),
                                 self._scale_y_pos(self.height/2. + rad*np.cos(theta)))
                )
                circle_poly.add_attr(circle_trans)
                self.bc_viewer.add_geom(circle_poly)
                ring_list.append(circle_poly)
            self.bc_polys.append(ring_list)

        # TODO: fill out this function correctly, for now just something basic
        #       to make sure the screen is working

    def _render_boundary_cell(self):

        if self.bc_viewer is None:
            self._create_boundary_cell_viewer()

        for r in range(self.bc_activations.shape[0]):
            for th in range(self.bc_activations.shape[1]):

                self.bc_polys[r][th].set_color(0, 0, self.bc_activations[r, th])

        return self.bc_viewer.render()
    
    def _create_grid_cell_viewer(self):
        """
        Viewer for grid cell activation
        """
        # TODO: should height and width be different than environment?
        self.gc_viewer = rendering.Viewer(self.screen_width, self.screen_height)

        # TODO: fill out this function correctly, for now just something basic
        #       to make sure the screen is working

    def _render_grid_cell(self):

        if self.gc_viewer is None:
            self._create_grid_cell_viewer()

        # TODO:

        return self.gc_viewer.render()
    
    def _create_place_cell_viewer(self):
        """
        Viewer for place cell activation
        """
        # TODO: should height and width be different than environment?
        self.pc_viewer = rendering.Viewer(self.screen_width, self.screen_height)

        # TODO: fill out this function correctly, for now just something basic
        #       to make sure the screen is working

    def _render_place_cell(self):

        if self.pc_viewer is None:
            self._create_place_cell_viewer()

        # TODO:

        return self.pc_viewer.render()

    def _render_extras(self):
        """
        Render all extra viewers
        includes: head direction, boundary cell, place cell, grid cell
        """

        if self.hd_activations is not None:
            self._render_head_direction()
        if self.bc_activations is not None:
            self._render_boundary_cell()
        #self._render_grid_cell()
        #self._render_place_cell()


    def _scale_x_pos(self, x):
        """
        Convert x coordinate to screen position for rendering
        """
        return x * self.scale + self.tile_size / 2.0

    def _scale_y_pos(self, y):
        """
        Convert y coordinate to screen position for rendering
        Y-axis is flipped in pyglet
        """
        return self.screen_height - (y * self.scale + self.tile_size / 2.0)

    def _scale_theta(self, theta):
        """
        Transformation on theta for rendering
        """
        if not self.continuous:
            theta = self._dir_to_ang(theta)
            return -theta + 3 * np.pi / 2.0
        else:
            return -theta + 0 * np.pi / 2.0

    def _dir_to_ang(self, direction):
        # TODO: make sure these mappings are correct
        if direction == self.UP:
            return 0
        elif direction == self.RIGHT:
            return -np.pi / 2
        elif direction == self.LEFT:
            return np.pi / 2
        elif direction == self.DOWN:
            return np.pi#2*np.pi
        else:
            print("Invalid discrete direction of {0}".format(direction))
            return 0

    def set_goal(self, x, y):
        """
        Manually set the goal to be a specific x-y location
        Used primarily for recreating an environment trained on a fixed goal location
        """
        self.goal_state[0] = x
        self.goal_state[1] = y

        # Populate the goal array, in case the observations use this representation
        self.goal_array = np.zeros_like(self.map_array)
        self.goal_array[int(self.goal_state[0]), int(self.goal_state[1])] = 1

    def seed(self, seed):
        # TODO: actually use this
        self._seed = seed

    def render(self, mode='human', close=False):
        """
        For compatibility with both gym versions
        """
        return self._render(mode=mode, close=close)


def generate_obs_dict(params):
    """
    Creates the appropriate observation dictionary from a set of parameters
    """
    obs = OrderedDict()

    if params['full_map_obs']:
        obs['full_map'] = True
    if params['pob'] > 0:
        obs['pob_view'] = {'pob_size': params['pob']}
    if params['n_sensors'] > 0:
        max_sensor_dist = params['max_sensor_dist']
        normalize_sensors = params['normalize_dist_sensors']
        fov_rad = params['fov'] * np.pi / 180
        obs['dist_sensors'] = {
            'n_sensors': params['n_sensors'],
            'fov_rad': fov_rad,
            'max_dist': max_sensor_dist,
            'normalize': normalize_sensors,
        }
    if params['n_grid_cells'] > 0:
        obs['grid_cell'] = {
            'n_grid_cells': params['n_grid_cells'],
            'grid_angle': params['grid_angle'],
            'grid_scale_range': params['grid_scales'],
            'gc_loc': None,  # TODO
            'gc_scale': None,  # TODO
        }
    if params['heading'] != 'none':
        obs['heading'] = {'circular': False, 'normalize': False, 'map_loc': False}
        if params['heading'] == 'map_loc':
            obs['heading']['map_loc'] = True
        elif params['heading'] == 'circular':
            obs['heading']['circular'] = True
        elif params['heading'] == 'normalized_angle':
            obs['heading']['normalize'] = True
    if params['location'] != 'none':
        if params['location'] == 'map_loc':
            obs['location'] = {'normalize': False, 'map_loc': True}
        elif params['location'] == 'normalized':
            obs['location'] = {'normalize': True, 'map_loc': False}
        else:
            obs['location'] = {'normalize': False, 'map_loc': False}
    if params['goal_loc'] != 'none':
        if params['goal_loc'] == 'map_loc':
            obs['goal_loc'] = {'normalize': False, 'map_loc': True}
        if params['goal_loc'] == 'normalized':
            obs['goal_loc'] = {'normalize': True, 'map_loc': False}
        else:
            obs['goal_loc'] = {'normalize': False, 'map_loc': False}
    if params['goal_vec'] != 'none':
        if params['goal_vec'] == 'normalized':
            obs['goal_vec'] = {'normalize': True}
        else:
            obs['goal_vec'] = {'normalize': False}
    if params['bc_n_ring'] > 0:
        # TODO: add option for random distribution in the future
        obs['boundary_cell'] = {
            'n_ring': params['bc_n_ring'],
            'n_rad': params['bc_n_rad'],
            'dist_rad': params['bc_dist_rad'],
            'receptive_field_min': params['bc_receptive_field_min'],
            'receptive_field_max': params['bc_receptive_field_max'],
        }
    if params['hd_n_cells'] > 0:
        # TODO: add option for random distribution in the future
        obs['hd_cell'] = {
            'n_cells': params['hd_n_cells'],
            'receptive_field_min': params['hd_receptive_field_min'],
            'receptive_field_max': params['hd_receptive_field_max'],
        }
    if params['csp_dim'] > 0:
        # TODO: make sure the unitary vectors created here are unique and controlled by the seed
        # TODO: have an option to supply specific vectors in the future NOTE: expecting SP rather than numpy array
        if 'x_axis_vec' in params:
            x_axis_vec = params['x_axis_vec']
        else:
            x_axis_vec = csp_utils.unitary_vector(params['csp_dim'])
        if 'y_axis_vec' in params:
            y_axis_vec = params['y_axis_vec']
        else:
            y_axis_vec = csp_utils.unitary_vector(params['csp_dim'])
    if params['goal_csp']:
        obs['goal_csp'] = {
            'egocentric': params['goal_csp_egocentric'],
            'dim': params['csp_dim'],
            'x_axis_vec': x_axis_vec,
            'y_axis_vec': y_axis_vec,
        }
    if params['agent_csp']:
        obs['agent_csp'] = {
            'dim': params['csp_dim'],
            'x_axis_vec': x_axis_vec,
            'y_axis_vec': y_axis_vec,
        }

    return obs
