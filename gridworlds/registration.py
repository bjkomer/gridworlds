from gym.envs.registration import registry, register, make, spec
from gridworlds.envs import GridWorldEnv, generate_obs_dict
import numpy as np

# Generate Gym Environment tags with predefined parameters

env_configs = {}

name_base = 'GW-{0}-{1}-{2}-v0'

# Set random seed to ensure environment generation is always consistent
np.random.seed(13)

# NOTE: currently using a fixed map, need to have a version with variable maps in the future
# Hardcoding here so the same map is used for the same environment specification, for reproducability
map_array = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

for continuous in [True, False]:
    for movement_type in ['directional', 'holonomic']:
        for biosensors in [True, False]:
            name = name_base.format(
                'Cont' if continuous else 'Disc',
                'Dir' if movement_type == 'directional' else 'Hol',
                'Bio' if biosensors else 'Basic',
            )

            base_params = {
                'full_map_obs': False,
                'pob': 0,
                'max_sensor_dist': 10,
                'n_sensors': 10,
                'fov': 180,
                'normalize_dist_sensors': True,
                'n_grid_cells': 0,
                'bc_n_ring': 0,
                'bc_n_rad': 0,
                'bc_dist_rad': 0,
                'bc_receptive_field_min': 0,
                'bc_receptive_field_max': 0,
                'hd_n_cells': 0,
                'hd_receptive_field_min': 0,
                'hd_receptive_field_max': 0,
                'csp_dim': 0,
                'goal_csp': False,
                'agent_csp': False,
                'goal_csp_egocentric': False,
            }

            if biosensors:
                specific_params = {
                    'heading': 'none',
                    'location': 'none',
                    'goal_loc': 'none',
                    'bc_n_ring': 12,
                    'bc_n_rad': 3,
                    'bc_dist_rad': .75,
                    'bc_receptive_field_min': 1,
                    'bc_receptive_field_max': 1.5,
                    'hd_n_cells': 8,
                    'hd_receptive_field_min': np.pi / 4,
                    'hd_receptive_field_max': np.pi / 4,
                    'goal_vec': 'normalized',
                    # TODO: add grid cell observations when implemented
                }
            else:
                specific_params = {
                    'location': 'normalized',
                    'goal_vec': 'normalized',
                    'goal_loc': 'normalized',
                    'heading': 'circular',
                }

            # Merge dictionaries, replacing base params with specific params
            params = {**base_params, **specific_params}

            obs_dict = generate_obs_dict(params)

            config = {
                'map_array': map_array,
                'observations': obs_dict,
                'continuous': continuous,
                'movement_type': movement_type,
                'dt': 0.1,
                'max_steps': 1000,
            }

            env_configs[name] = config

simple_map_array = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

# Simple CSP environment

params = {
    'full_map_obs': False,
    'pob': 0,
    'max_sensor_dist': 0,
    'n_sensors': 0,
    'fov': 0,
    'normalize_dist_sensors': False,
    'n_grid_cells': 0,
    'bc_n_ring': 0,
    'bc_n_rad': 0,
    'bc_dist_rad': 0,
    'bc_receptive_field_min': 0,
    'bc_receptive_field_max': 0,
    'hd_n_cells': 0,
    'hd_receptive_field_min': 0,
    'hd_receptive_field_max': 0,
    'heading': 'none',
    'location': 'none',
    'goal_loc': 'none',
    'goal_vec': 'none',
    'goal_csp': True,
    'agent_csp': False,
    'goal_csp_egocentric': True,
    'csp_dim': 256,
}

obs_dict = generate_obs_dict(params)

env_configs['GW-CSP-Simple-v0'] = {
    'map_array': simple_map_array,
    'observations': obs_dict,
    'continuous': True,
    'movement_type': 'holonomic',
    'dt': 0.1,
    'max_steps': 1000,
}

env_configs['GW-CSP-Disc-Simple-v0'] = {
    'map_array': simple_map_array,
    'observations': obs_dict,
    'continuous': False,
    'movement_type': 'holonomic',
    'dt': 0.1,
    'max_steps': 20,
}

simple_small_map_array = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
])

env_configs['GW-CSP-Simple-Small-v0'] = {
    'map_array': simple_small_map_array,
    'observations': obs_dict,
    'continuous': True,
    'movement_type': 'holonomic',
    'dt': 0.1,
    'max_steps': 100,
}

env_configs['GW-CSP-FixedEpisode-Simple-Small-v0'] = {
    'map_array': simple_small_map_array,
    'observations': obs_dict,
    'continuous': True,
    'movement_type': 'holonomic',
    'fixed_episode_length': True,
    'dt': 0.1,
    'max_steps': 100,
}

env_configs['GW-CSP-FixedEpisode-Simple-v0'] = {
    'map_array': simple_map_array,
    'observations': obs_dict,
    'continuous': True,
    'movement_type': 'holonomic',
    'fixed_episode_length': True,
    'dt': 0.1,
    'max_steps': 100,
}

# Various simple environments for debugging
params_debug = {
    'full_map_obs': False,
    'pob': 0,
    'max_sensor_dist': 0,
    'n_sensors': 0,
    'fov': 0,
    'normalize_dist_sensors': False,
    'n_grid_cells': 0,
    'bc_n_ring': 0,
    'bc_n_rad': 0,
    'bc_dist_rad': 0,
    'bc_receptive_field_min': 0,
    'bc_receptive_field_max': 0,
    'hd_n_cells': 0,
    'hd_receptive_field_min': 0,
    'hd_receptive_field_max': 0,
    'heading': 'none',
    'location': 'none',
    'goal_loc': 'actual',
    'goal_vec': 'none',
    'goal_csp': False,
    'agent_csp': False,
    'goal_csp_egocentric': False,
    'csp_dim': 0,
}
obs_dict_debug = generate_obs_dict(params_debug)

env_configs['GW-Debug-Simple-v0'] = {
    'map_array': simple_map_array,
    'observations': obs_dict_debug,
    'continuous': True,
    'movement_type': 'holonomic',
    'dt': 0.1,
    'max_steps': 100,
    'fixed_episode_length': True,
}

for name, kwargs in env_configs.items():
    register(
        id=name,
        entry_point='gridworlds.envs:GridWorldEnv',
        kwargs=kwargs,
        # tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
        max_episode_steps=1000,
        nondeterministic=False,
    )
