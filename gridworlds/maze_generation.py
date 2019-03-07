import numpy as np
from scipy.ndimage import measurements

from gym_maze.envs.generators import SimpleMazeGenerator, RandomMazeGenerator, \
        RandomBlockMazeGenerator, TMazeGenerator, WaterMazeGenerator


def generate_maze(map_style='blocks', side_len=10, obstacle_ratio=.2):
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
