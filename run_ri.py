import numpy as np
import gym
import random
import time
from uofgsocsai import LochLomondEnv
from helpers import *


def run(pid, map):

    # env = gym.make('FrozenLake-v0')
    reward_hole = 0.0

    env = LochLomondEnv(problem_id=pid, is_stochastic=False, reward_hole=reward_hole, map_name_base=map)
    env.render()
    # state_space, action_space, state_initial_id, state_goal_id = env2statespace(env)

    action_space = env.action_space.n
    state_space = env.observation_space.n

    q_table = np.zeros((state_space, action_space))

    # parameter set up
    num_episodes = 5000
    max_steps_per_episode = 1000
    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01

    rewards = []

    for episode in range(num_episodes):
        state = env.reset()

        done = False
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):

            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)
            # env.render()

            # update q table
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done == True:
                if rewards_current_episode>0:
                    print('you reached the goal in {} steps'.format(step))
                else:
                    print('you fell in {} steps'.format(step))
                break

        # exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        rewards.append(rewards_current_episode)

    rewards_per_1000_eps = np.split(np.array(rewards), num_episodes/1000)
    count = 1000
    print("***avg reward per 1000 episodes")
    for r in rewards_per_1000_eps:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    print("*** Q table ****")
    print(q_table)



if __name__=="__main__":

    map_name_base_8 = "8x8-base"
    map_name_base_4 = "4x4-base"

    problem_id= 0

    run(pid=problem_id, map=map_name_base_4)