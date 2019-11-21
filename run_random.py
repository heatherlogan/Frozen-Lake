# Imports
import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv  # load the class defining the custom Open AI Gym problem
import os, sys
from helpers import *

print("Working dir:" + os.getcwd())
print("Python version:" + sys.version)

goal_episode_counter = []
hole_episode_counter = []
goal_iteration_counter_average = []
goal_iteration_counter_bestcase = []
goal_iteration_counter_worstcase = []
first_goal_counter = []

def mean(array):
    if len(array)==0:
        return 0
    return np.mean(array)

def mini(array):
    if len(array)==0:
        return 0
    return np.min(array)

def maxi(array):
    if len(array)==0:
        return 0
    return np.max(array)

def run_senseless_agent(number_of_repeats, problem_id, map):

    for i in range(number_of_repeats):

        problem_id = problem_id
        reward_hole = 0.0
        is_stochastic = True

        max_episodes = 0000
        max_iter_per_episode = 2000

        env = LochLomondEnv(problem_id=problem_id, is_stochastic=is_stochastic,
                            map_name_base=map,
                            reward_hole=reward_hole)

        env.action_space.sample()
        np.random.seed(12)

        # variables for performance evaluation

        # number of times goal is reached out of max_episodes/ (performance measures where reward is collected)
        goal_episodes = []
        # average number of iterations taken to reach goal per rewarded episode
        goal_iterations = []
        # number of episodes before goal is first reached
        first_goal = 0



        for e in range(max_episodes):
            observation = env.reset()

            for iter in range(max_iter_per_episode):

                #env.render()
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)

                #print("e,iter,reward,done =" + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done))

                if (done and reward == reward_hole):
                    #env.render()
                    #print("e,iter,reward,done =" + str(e) + " " + str(iter) + " " + str(reward) + " " + str(done))
                    #print("We have reached a hole")
                    hole_episode_counter.append(e)
                    break

                if (done and reward == +1.0):
                    #env.render()
                    goal_episodes.append(e)
                    goal_iterations.append(iter)

                    if first_goal == 0:
                        first_goal = e
                    break

                    # print("We have reached the goal in iteration ", e)
        # try:
        #     print()
        #     print('-----Rep {}------'.format(i))
        #     print("Number of successful episodes ", len(goal_episodes))
        #     print("Average number of iterations in successful episodes ", mean(goal_iterations))
        #     print("Episodes before first goal ", first_goal)
        #
            goal_episode_counter.append(len(goal_episodes))
            goal_iteration_counter_average.append(mean(goal_iterations))
            goal_iteration_counter_bestcase.append(mini(goal_iterations))
            goal_iteration_counter_worstcase.append(maxi(goal_iterations))
            first_goal_counter.append(first_goal)
        # except ValueError:
        #     pass

    try:
        print()
        print()
        print('-----Final stats for {} repetitions on Problem {}-----'.format(number_of_repeats, problem_id))
        print("Average Number of successful episodes out of 10000:", mean(goal_episode_counter))
        print("Average Number of unsuccessful episodes out of 10000:", mean(hole_episode_counter))
        print("Average number of iterations in successful episodes: ", mean(goal_iteration_counter_average))
        print("Avg best-case number of iterations in successful episodes: ", mean(goal_iteration_counter_bestcase))
        print("Avg worst-case number of iterations in successful episodes: ", mean(goal_iteration_counter_worstcase))
        print("Average Episodes before first goal ", mean(first_goal_counter))
    except ValueError:
        pass

    return [mean(goal_episode_counter), mean(hole_episode_counter),
            mean(goal_iteration_counter_average), mean(goal_iteration_counter_bestcase),
            mean(goal_iteration_counter_worstcase), mean(first_goal_counter) ]

if __name__=="__main__":

    file = open('output_tables/senseless_agent_4x4.csv', 'w')

    map_name_base_8 = "8x8-base"
    map_name_base_4 = "4x4-base"

    file.write("problem_id, avg rewards in 10000 eps, avg failures in 10000 eps, avg iterations to reward, avg best-case iterations to reward, avg worst case iterations to reward\n")

    for i in range(0,8):
        a,b,c,d,e,f = run_senseless_agent(number_of_repeats=5, problem_id=i, map=map_name_base_4)
        file.write("problem_{}, {}, {},{},{},{},{}\n".format(i,a,b,c,d,e,f))


    file.close()