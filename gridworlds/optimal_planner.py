import numpy as np

class OptimalPlanner(object):
    """
    Given full information in a GridWorldEnv, compute actions that will take the agent
    directly to the goal. To be used as a baseline for optimal performance as well as
    an oracle for imitation learning
    TODO: should this just be an optional function of the GridWorldEnv?
    """

    def __init__(self):
        self.actions = None
        self.t = 0

    def form_plan(self, env):
        """
        Create a plan of actions by simulating their effect on a
        deterministic environment.
        In the continuous case, first generate the discrete trajectory that solves
        the maze, and then generate continuous commands to follow that trajectory
        env will also contain the current state information and the goal
        """

        # Reset t for the new plan
        self.t = 0

        # # Compute the optimal path using discrete grids
        # coarse_plan = self.generate_coarse_plan(env)
        #
        # # Generate a list of coordinates corresponding to the center of those grid points
        # target_points = self.generate_target_points(coarse_plan)

        target_points = self.generate_coarse_plan(env)

        # Generate actions of a controller to move to those points
        self.actions = self.generate_actions(env, target_points)

        # self.t = 1
        # return self.actions[0]

    def generate_coarse_plan(self, env, wall_value=1000):
        """
        use simple wavefront algorithm to find the shortest path
        movement cost of 1 for sides, and 1.42 for diagonals
        only allow diagonal movement if both sides are open space
        this may not be the most efficient method, but most maps
        are relatively small, so it will be fine
        """

        # Copy over the map, and make all the walls some large number
        planning_grid = wall_value * env.map_array.copy()

        start_state = np.array([int(np.round(env.state[0])), int(np.round(env.state[1]))])
        goal_state = np.array([int(np.round(env.goal_state[0])), int(np.round(env.goal_state[1]))])

        # start with the goal state
        to_expand = [
            (goal_state[0], goal_state[1])
        ]

        planning_grid[goal_state[0], goal_state[1]] = 1

        finished = False

        while len(to_expand) > 0 and not finished:
            next_node = to_expand.pop()
            new_nodes = self.expand_node(planning_grid, next_node)
            to_expand += new_nodes
            # # Check to see if expansion got back to the start
            # if planning_grid[start_state[0], start_state[1]] != 0:
            #     finished = True

        # Now follow the shortest path from start to goal
        target_points = []

        current_point = (start_state[0], start_state[1])

        while current_point != (goal_state[0], goal_state[1]):
            section = planning_grid[current_point[0]-1: current_point[0]+2, current_point[1]-1: current_point[1]+2].copy()

            # Fill in undesired diagonal motions
            if section[0, 1] == wall_value:
                section[0, :] = wall_value
            if section[1, 0] == wall_value:
                section[:, 0] = wall_value
            if section[2, 1] == wall_value:
                section[2, :] = wall_value
            if section[1, 2] == wall_value:
                section[:, 2] = wall_value

            ind = np.argmin(section)
            # TODO: make sure x-y are not flipped
            dx = ind // 3 - 1
            dy = ind % 3 - 1
            if dx == 0 and dy == 0:
                print("Anamoly detected in optimal path algorithm")
                print(ind)
                print(current_point)
                print(planning_grid[current_point[0]-1: current_point[0]+2, current_point[1]-1: current_point[1]+2])
                raise RuntimeError
            current_point = (current_point[0] + dx, current_point[1] + dy)
            target_points.append(current_point)
            # next_point = self.get_next_point(planning_grid, current_point)
            # target_points.append(next_point)

        return target_points

    def expand_node(self, planning_grid, node, diagonal_cost=1.42, wall_value=1000):

        current_value = planning_grid[node[0], node[1]]

        # Generate list of indices for all nodes around the current node
        # NOTE: shouldn't need a bounds check since the edge of all mazes is walls

        checks = [
            (node[0] + 1, node[1], current_value + 1),
            (node[0] - 1, node[1], current_value + 1),
            (node[0], node[1] + 1, current_value + 1),
            (node[0], node[1] - 1, current_value + 1),
        ]

        diag_checks = [
            (node[0] + 1, node[1] + 1, current_value + diagonal_cost),
            (node[0] - 1, node[1] + 1, current_value + diagonal_cost),
            (node[0] + 1, node[1] - 1, current_value + diagonal_cost),
            (node[0] - 1, node[1] - 1, current_value + diagonal_cost),
        ]

        new_nodes = []

        for c in checks:
            x, y, value = c
            if planning_grid[x, y] != wall_value and (planning_grid[x, y] == 0 or planning_grid[x, y] > value):
                planning_grid[x, y] = value
                new_nodes.append((x, y))

        # handle diagonals differently, only allow passage if both edges are also free
        for c in diag_checks:
            x, y, value = c
            if planning_grid[x, y] != wall_value and (planning_grid[x, y] == 0 or planning_grid[x, y] > value):
                # extra check for diagonals
                if planning_grid[x, node[1]] != wall_value and planning_grid[node[0], y] != wall_value:
                    planning_grid[x, y] = value
                    new_nodes.append((x, y))

        return new_nodes

    # def get_next_point(self, planning_grid, node):
    #
    #     checks = [
    #         (node[0] + 1, node[1]),
    #         (node[0] - 1, node[1]),
    #         (node[0] + 1, node[1] + 1),
    #         (node[0] - 1, node[1] + 1),
    #         (node[0] + 1, node[1] - 1),
    #         (node[0] - 1, node[1] - 1),
    #         (node[0], node[1] + 1),
    #         (node[0], node[1] - 1),
    #     ]
    #
    #     value = 1000
    #     for x, y in zip(checks):

    def generate_target_points(self, coarse_plan):
        raise NotImplementedError

    def generate_actions(self, env, target_points):

        state = env.state.copy()

        # Target point index
        ti = 0

        actions = []

        while np.linalg.norm(state[[0, 1]] - env.goal_state[[0, 1]]) > .5:
            # Compute the next action and apply it to the simulated state
            state, action = self.controller_step(state, target=target_points[ti], env=env)
            actions.append(action)

            # If close enough to a target point, select the next point as the new target
            if np.linalg.norm(state[[0, 1]] - target_points[ti]) < .9:
                ti += 1
                ti = min(ti, len(target_points) - 1)

        return np.array(actions)

    def controller_step(self, state, target, env):
        """
        Perform one step of the controller, and return the new state and the action that got there
        """
        # TODO: assuming continuous directional control, will need to implement the other methods still

        th = state[2]
        dx = target[0] - state[0]
        dy = target[1] - state[1]
        desired_th = np.arctan2(dy, dx)
        dth = desired_th - th
        if dth > np.pi:
            dth -= 2*np.pi
        if dth < -np.pi:
            dth += 2*np.pi
        dlin = np.sqrt(dx**2+dy**2)

        if abs(dth) < np.pi/2:
            lin_ac = np.clip(dlin, 0, env.max_lin_vel)
        else:
            lin_ac = 0
        ang_ac = np.clip(dth, -env.max_ang_vel, env.max_ang_vel)

        state[2] += ang_ac * env.dt
        if state[2] > np.pi:
            state[2] -= 2 * np.pi
        elif state[2] < -np.pi:
            state[2] += 2 * np.pi
        displacement = np.array([np.cos(state[2]), np.sin(state[2])]) * lin_ac * env.dt

        state[[0, 1]] += displacement

        return state, np.array([lin_ac, ang_ac])

    def next_action(self):
        self.t += 1
        if len(self.actions) == 0:
            print("Warning: No actions to take")
            return np.zeros(2)
        if self.t > len(self.actions):
            # raise RuntimeError("Attempting to take an action on a completed trajectory")
            print("Warning: Taking an action on a completed trajectory")
            return self.actions[-1]
        return self.actions[self.t - 1]

    def act(self, obs, env):

        # observations are not needed here with full environment state.
        # they are just included as an argument so the function signature
        # is consistent with other expert policies, which may use observations

        self.form_plan(env)

        return self.next_action()

    def __getitem__(self, index):

        return self.actions[index]
