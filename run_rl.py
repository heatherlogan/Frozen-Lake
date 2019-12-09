import numpy as np
import pandas as pd
import sys
import gym
import random
import time
from uofgsocsai import LochLomondEnv
from helpers import *
from run_random import mini, maxi, mean


def run_reinforcement_agent(problem_id, map):
    reward_hole = -0.5

    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole, map_name_base=map)
    env.reset()

    # state_space, action_space, state_initial_id, state_goal_id = env2statespace(env)

    action_space = env.action_space.n
    state_space = env.observation_space.n

    q_table = np.zeros((state_space, action_space))

    # parameter set up
    max_episodes = 10000
    iterations = 1000
    learning_rate = 0.1  # alpha
    discount_rate = 0.95  # gamma
    epsilon = 0.05 # exploration-exploitation settup
    rewards = []

    hole_episode_counter = []

    # number of times goal is reached out of max_episodes/ (performance measures where reward is collected)
    goal_episodes = []
    # average number of iterations taken to reach goal per rewarded episode
    goal_iterations = []
    # number of episodes before goal is first reached
    first_goal = 0

    for episode in range(max_episodes):
        state = env.reset()

        # end learning phase at midpoint
        if episode == max_episodes / 2:
            # print('LEARNING OVER')
            learning_rate = 0.0

        rewards_current_episode = 0

        for step in range(iterations):

            # choose the highest q_value in table to choose action

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            # if q_table is empty, random choice
            if action == 0:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            # update q table
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) \
                                     + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done == True:
                if rewards_current_episode != reward_hole:
                    # set first episode that goal is reached
                    if first_goal == 0:
                        first_goal = episode

                    goal_episodes.append(episode)
                    goal_iterations.append(step+2)
                    # print('you reached the goal in {} steps'.format(step))
                    break
                else:
                    # print('you fell in {} steps'.format(step))
                    hole_episode_counter.append(episode)

                    break

        rewards.append(rewards_current_episode)

    rewards_per_100_eps = np.split(np.array(rewards), max_episodes / 100)
    rewards_per_100_eps = [str(sum(r / 100)) for r in rewards_per_100_eps]

    return len(goal_episodes), len(hole_episode_counter), mean(goal_iterations), \
           mini(goal_iterations), maxi(goal_iterations), first_goal, rewards_per_100_eps


if __name__ == "__main__":
    problem_id = sys.argv[1]
    map_name = sys.argv[2]

    all_results = []

    results = run_reinforcement_agent(int(problem_id), map_name)

    idx = ['Total Rewards in 10000 eps', 'Total Failures in 10000 eps', 'Avg Steps to Goal', 'Best-case steps to Goal',
           'Worst-case steps to Goal', 'Episodes to first Goal']
    df = pd.DataFrame((results[:-1]), columns=['problem_{}'.format(problem_id)])
    df['Measure'] = idx
    df = df.set_index('Measure')
    print('-------------REINFORCEMENT AGENT--------------\n', df)
