import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches


class ObservationViewer(object):

    supported_observations = [
        # 'full_map',
        # 'pob_view',
        # 'dist_sensors',
        # 'heading',
        # 'location',
        # 'grid_cells',
        # 'goal_loc',
        'goal_vec',
        'boundary_cell',
        'hd_cell',
        # 'goal_csp',
        # 'agent_csp',
    ]

    def __init__(self, obs_dict, obs_index_dict, title=None):

        # Specification for the type of observations used
        self.obs_dict = obs_dict

        # Mapping of elements from the observation vector to the appropriate observation modality
        self.obs_index_dict = obs_index_dict

        # String names for each observation modality
        self.modalities = []
        # Only keep track of observations that have a corresponding plot
        for obs in self.obs_index_dict.keys():
            if obs in self.supported_observations:
                self.modalities.append(obs)

        # The number of things to plot
        self.n_plots = len(self.modalities)

        # Generate a figure with subplots arranged in a square
        # Save a reference to the subplot indexed by the observation name

        n_row = int(np.ceil(np.sqrt(self.n_plots)))
        n_col = int(np.ceil(self.n_plots / n_row))

        # NOTE: don't really need labels for any of these plots
        self.fig, self.ax = plt.subplots(
            n_row, n_col,
            # sharex='col', sharey='row'
        )

        if title is not None:
            self.fig.suptitle(title)

        # Build a dictionary mapping to the axes
        self.ax_dict = {}
        i = 0
        for r in range(n_row):
            for c in range(n_col):
                self.ax_dict[self.modalities[i]] = self.ax[r, c]
                i += 1
                if i >= self.n_plots:
                    break
            if i >= self.n_plots:
                break

        self.obs = None

        self.init_figures()

    def update(self, obs):

        self.update_obs(obs)
        self.update_figures()

    def update_obs(self, obs):

        self.obs = obs

    def update_figures(self):

        for obs in self.ax_dict.keys():
            if obs == 'boundary_cell':
                self._boundary_cell()
            if obs == 'hd_cell':
                self._hd_cell()
            if obs == 'goal_vec':
                self._goal_vec()

        self.fig.canvas.draw()

    def init_figures(self):

        for obs in self.ax_dict.keys():
            if obs == 'boundary_cell':
                self._init_boundary_cell()
            if obs == 'hd_cell':
                self._init_hd_cell()
            if obs == 'goal_vec':
                self._init_goal_vec()

        self.fig.canvas.draw()

    def _init_boundary_cell(self):
        n_ring = self.obs_dict['boundary_cell']['n_ring']
        n_rad = self.obs_dict['boundary_cell']['n_rad']
        dist_rad = self.obs_dict['boundary_cell']['dist_rad']

        limit = n_rad * dist_rad + 0.5
        self.ax_dict['boundary_cell'].set_xlim(-limit, limit)
        self.ax_dict['boundary_cell'].set_ylim(-limit, limit)

        self.bc_circles = np.empty((n_ring * n_rad,), dtype=object)

        for r in range(n_rad):
            for th in range(n_ring):
                rad = (r + 1) * dist_rad
                theta = th * 2 * np.pi / n_ring

                x = rad*np.sin(theta)
                y = rad*np.cos(theta)

                # Save the object in an array accessible by the flattened indices
                self.bc_circles[r * n_ring + th] = patches.Circle(xy=(x, y), radius=.3)

                self.ax_dict['boundary_cell'].add_artist(self.bc_circles[r * n_ring + th])

    def _boundary_cell(self):
        data = self.obs[self.obs_index_dict['boundary_cell']]

        for i, val in enumerate(data):

            b = max(0, min(1, val))

            self.bc_circles[i].set_facecolor((0, 0, b))

    def _init_hd_cell(self):
        n_cells = self.obs_dict['hd_cell']['n_cells']

        self.ax_dict['hd_cell'].set_xlim(-1.5, 1.5)
        self.ax_dict['hd_cell'].set_ylim(-1.5, 1.5)

        self.hd_circles = np.empty((n_cells,), dtype=object)

        for i in range(n_cells):

            #FIXME: NOTE: adding an extra 90 degrees to align with the viewer
            #             currently looking to the right is 0 degrees in the code
            theta = -i * 2*np.pi / n_cells - np.pi/2.
            x = 1*np.sin(theta)
            y = 1*np.cos(theta)
            self.hd_circles[i] = patches.Circle(xy=(x, y), radius=.3)

            self.ax_dict['hd_cell'].add_artist(self.hd_circles[i])

    def _hd_cell(self):
        data = self.obs[self.obs_index_dict['hd_cell']]

        for i, val in enumerate(data):

            b = max(0, min(1, val))

            self.hd_circles[i].set_facecolor((0, 0, b))

    def _init_goal_vec(self):
        normalize = self.obs_dict['goal_vec']['normalize']

        self.goal_vec_arrow = patches.Arrow(x=0, y=0, dx=-1, dy=1)
        self.ax_dict['goal_vec'].set_xlim(-2, 2)
        self.ax_dict['goal_vec'].set_ylim(-2, 2)
        self.goal_vec_arrow_handle = self.ax_dict['goal_vec'].add_artist(self.goal_vec_arrow)

    def _goal_vec(self):
        data = self.obs[self.obs_index_dict['goal_vec']]

        self.goal_vec_arrow_handle.remove()
        self.goal_vec_arrow = patches.Arrow(x=0, y=0, dx=data[0], dy=data[1])
        self.goal_vec_arrow_handle = self.ax_dict['goal_vec'].add_artist(self.goal_vec_arrow)





