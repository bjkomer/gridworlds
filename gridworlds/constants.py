import numpy as np

# Action constants for discrete environment
FORWARD = 0
UP = 0
LEFT = 2  # 1
RIGHT = 1  # 2
DOWN = 3

# State transition mappings for discrete environments
# given movement_action, compute displacement
holonomic_transitions = {
    UP: np.array([0, 1]),
    DOWN: np.array([0, -1]),
    LEFT: np.array([-1, 0]),
    RIGHT: np.array([1, 0]),
}

# given (direction_action, current_heading) produce (next_heading)
directional_transitions = {
    (LEFT, UP): LEFT,
    (RIGHT, UP): RIGHT,
    (LEFT, DOWN): RIGHT,
    (RIGHT, DOWN): LEFT,
    (LEFT, LEFT): DOWN,
    (RIGHT, LEFT): UP,
    (LEFT, RIGHT): UP,
    (RIGHT, RIGHT): DOWN,
}

possible_objects = [
    'watermelon',
    'strawberry',
    'plum',
    'orange',
    'onion',
    'nut',
    'mushroom',
    'gingerbread-house',
    'corn',
    'coconut',
    'cherry',
    'bacon',
    'avocado',
]
