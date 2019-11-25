import numpy as np
from scipy.ndimage import measurements

from gym_maze.envs.generators import SimpleMazeGenerator, RandomMazeGenerator, \
        RandomBlockMazeGenerator, TMazeGenerator, WaterMazeGenerator


def generate_maze(map_style='blocks', side_len=10, obstacle_ratio=.2, node_radius=5, corridor_width=5, node_range=(5, 10)):
    if map_style == 'maze':
        # NOTE: only allows odd shapes
        maze = RandomMazeGenerator(width=side_len - 2,
                                   height=side_len - 2,
                                   complexity=.75, density=.75)
    elif map_style == 'blocks':
        maze = RandomBlockMazeGenerator(maze_size=side_len - 2,  # the -2 is because the outer wall gets added
                                        obstacle_ratio=obstacle_ratio,
                                       )
    elif map_style == 't':
        raise NotImplementedError
        # TODO: figure out what the parameters of the T maze should be to get an appropriate size maze
        maze = TMazeGenerator()
    elif map_style == 'morris':
        maze = WaterMazeGenerator(radius_maze=int(side_len / 2.),
                           radius_platform=4)
    elif map_style == 'simple':
        raise NotImplementedError
        # TODO: this seems to be expecting a custom layout
        maze = SimpleMazeGenerator()
    elif map_style == 'empty' or map_style == 'empty-static':
        # Fill with empty space
        maze = np.zeros((side_len, side_len))

        # Place walls along the edges
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1

        return maze
    elif map_style == 'corridors':
        maze = random_corridors(
            side_len=side_len,
            node_radius=node_radius, corridor_width=corridor_width,
            node_range=node_range
        )

        return remove_inaccessible_space(maze)

    return remove_inaccessible_space(maze.maze)


def remove_inaccessible_space(maze):
    """
    Leave only the largest fully accessible region intact
    Fill all other holes with walls
    """

    # Generate an array with the contiguous clusters labelled
    # also returns the number of clusters
    label, num_features = measurements.label(1 - maze)

    # Find the largest cluster
    largest_class = -1
    largest_count = -1
    for i in range(1, num_features + 1):
        count = (label == i).sum()
        if count > largest_count:
            largest_count = count
            largest_class = i

    # Make a new maze that is walls everywhere except for that cluster
    new_maze = np.ones_like(maze)
    new_maze -= (label == largest_class)

    return new_maze


def random_corridors(side_len, node_radius, corridor_width, node_range, rng=np.random):

    # The nodes should not be bigger than the area to work with
    assert side_len > 2 * node_radius
    # Requiring at least 4 nodes
    assert node_range[0] >= 4

    maze = np.ones((side_len, side_len))

    # choose a random number of nodes within the specified range
    n_nodes = rng.randint(node_range[0], node_range[1] + 1)

    # choose random node locations within the area
    node_centers = rng.uniform(low=node_radius, high=side_len - node_radius, size=(n_nodes, 2))

    # Force the first four nodes to be in different quadrants, to ensure the map is spaced out
    min_val = node_radius
    max_val = side_len - node_radius
    range_val = max_val - min_val
    node_centers[0, 0] = rng.uniform(low=min_val, high=min_val + range_val / 3)
    node_centers[0, 1] = rng.uniform(low=min_val, high=min_val + range_val / 3)

    node_centers[1, 0] = rng.uniform(low=min_val, high=min_val + range_val / 3)
    node_centers[1, 1] = rng.uniform(low=max_val - range_val / 3, high=max_val)

    node_centers[2, 0] = rng.uniform(low=max_val - range_val / 3, high=max_val)
    node_centers[2, 1] = rng.uniform(low=max_val - range_val / 3, high=max_val)

    node_centers[3, 0] = rng.uniform(low=max_val - range_val / 3, high=max_val)
    node_centers[3, 1] = rng.uniform(low=min_val, high=min_val + range_val / 3)

    # randomly connect nodes together, with the probability based on how close in space they are
    # closer nodes are more likely to have a connection

    connectivity = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if i != j:
                dist = np.linalg.norm(node_centers[i, :] - node_centers[j, :])
                if rng.uniform(low=0, high=side_len*np.sqrt(2)) < dist:
                    # connect the nodes
                    connectivity[i, j] = 1

    # place free space into the environment
    for i in range(n_nodes):
        # free space of the node
        mask = node(center=node_centers[i, :], radius=node_radius, side_len=side_len)
        maze[mask] = 0
        for j in range(i + 1, n_nodes):
            if connectivity[i, j] == 1:
                # draw a connection between the nodes
                mask = line(p1=node_centers[i, :], p2=node_centers[j, :], width=corridor_width, side_len=side_len)
                maze[mask] = 0

    # ensure the edges are all walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    return maze


def node(center, radius, side_len):
    xs = np.arange(side_len)
    arr = np.zeros((side_len, side_len))

    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                arr[i, j] = 1

    # return a mask
    return np.where(arr == 1)


def line(p1, p2, width, side_len):
    xs = np.arange(side_len)
    arr = np.zeros((side_len, side_len))

    dist = np.linalg.norm(p2 - p1)
    n_segments = int(np.ceil(dist)*2)

    segment_centers = np.zeros((n_segments, 2))

    segment_centers[0, :] = p1

    slope = (p2 - p1) / n_segments

    for s in range(1, n_segments):
        segment_centers[s, :] = segment_centers[s - 1, :] + slope

    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            # draw many circles along the line
            # this is a hacky and slow way to do it, but should work alright
            for s in range(n_segments):
                if (x - segment_centers[s, 0])**2 + (y - segment_centers[s, 1])**2 < (width/2)**2:
                    arr[i, j] = 1

    # return a mask
    return np.where(arr == 1)
