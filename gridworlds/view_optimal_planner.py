import numpy as np
import time
import gym
import argparse

from gridworlds.registration import env_configs
from gridworlds.optimal_planner import OptimalPlanner

parser = argparse.ArgumentParser("View the performance of a hand-crafted planner on a GridWorldEnv")
parser.add_argument('--env', default='GW-Cont-Dir-Bio-v0', choices=env_configs.keys())
args = parser.parse_args()

env = gym.make(args.env)

num_episodes = 10
returns = np.zeros((num_episodes,))

continuous = 'Cont' in args.env
directional = 'Dir' in args.env

planner = OptimalPlanner(continuous=continuous, directional=directional)

num_episodes = 10
time_steps = 1000

for e in range(num_episodes):
    obs = env.reset()
    # planner.form_plan(env)
    for s in range(time_steps):
        planner.form_plan(env.env)
        action = planner.next_action()
        #print(action)
        #action = env.action_space.sample()
        #print(action)
        #print("")
        obs, reward, done, info = env.step(action)
        # print(obs)
        returns[e] += reward
        # if reward != 0:
        #    print(reward)
        env.render()
        if not continuous:
            time.sleep(0.1)
        #time.sleep(env.env.dt)
        if done:
            break

print(returns)
