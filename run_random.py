# Imports
import gym
import numpy as np
import pandas as pd
import time
from uofgsocsai import LochLomondEnv  # load the class defining the custom Open AI Gym problem
import os, sys
from helpers import *


# helper functions to protect against zero divisions

def mean(array):
    if len(array) == 0:
        return 0
    return np.mean(array)


def mini(array):
    if len(array) == 0:
        return 0
    return np.min(array)


def maxi(array):
    if len(array) == 0:
        return 0
    return np.max(array)


def run_senseless_agent(problem_id, map):

    reward_hole = 0.0
    max_episodes = 10000
    max_iter_per_episode = 1000

    env = LochLomondEnv(problem_id=problem_id, is_stochastic=True,
                        map_name_base=map,
                        reward_hole=reward_hole)

    env.render()
    env.action_space.sample()

    np.random.seed(12)

    # variables for performance evaluation
    # number of times goal is reached out of max_episodes/ (performance measures where reward is collected)
    goal_episodes = []
    # number of episodes agent falls in hole
    hole_episodes = []
    # average number of iterations taken to reach goal per rewarded episode
    goal_iterations = []

    rewards = []

    # number of episodes before goal is first reached
    first_goal = 0

    for e in range(max_episodes):

        rewards_current_episode = 0
        state = env.reset()

        for iter in range(max_iter_per_episode):

            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            rewards_current_episode += reward

            if (done and reward == reward_hole):
                hole_episodes.append(e)
                break

            if (done and reward == +1.0):
                # env.render()
                goal_episodes.append(e)
                goal_iterations.append(iter+2)

                # sets first goal to episode
                if first_goal == 0:
                    first_goal = e
                break

        rewards.append(rewards_current_episode)

    # calculating steps to goal
    goal_iteration_average = mean(goal_iterations)
    goal_iteration_bestcase = mini(goal_iterations)
    goal_iteration_worstcase = maxi(goal_iterations)


    # splits collected rewards into per 100 episodes
    rewards_per_100_eps = np.split(np.array(rewards), max_episodes / 100)
    rewards_per_100_eps = [str(sum(r / 100)) for r in rewards_per_100_eps]


    return len(goal_episodes), len(hole_episodes), goal_iteration_average, goal_iteration_bestcase, \
           goal_iteration_worstcase,  first_goal, rewards_per_100_eps

if __name__ == "__main__":

    problem_id = sys.argv[1]
    map_name = sys.argv[2]

    results = run_senseless_agent(int(problem_id), map_name)

    idx= ['Total Rewards in 10000 eps', 'Total Failures in 10000 eps', 'Avg Steps to Goal', 'Best-case steps to Goal', 'Worst-case steps to Goal', 'Episodes to first Goal']
    df = pd.DataFrame((results[:-1]), columns=['problem_{}'.format(problem_id)])
    df['Measure'] = idx
    df = df.set_index('Measure')
    print('\n---------------RANDOM AGENT----------------\n', df)






