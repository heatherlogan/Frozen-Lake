from aima3.search import *
from uofgsocsai import LochLomondEnv
from helpers import *
import gym
import numpy as np

def run(pid, map):
    env = gym.make('FrozenLake8x8-v0')
    # env = LochLomondEnv(problem_id=pid, is_stochastic=True, reward_hole=-1.0, map_name_base=map)

    # if 8x8 then observation space is 64, else 16
    if map=="8x8-base":
        Q = np.zeros([64,4])
    else:
        Q = np.zeros([16, 4])

    # 2. Parameters of Q-leanring
    eta = .628
    gma = .9
    epis = 5000
    rev_list = []  # rewards per episode calculate
    # 3. Q-learning Algorithm
    for i in range(epis):
        # Reset environment
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        # The Q-Table learning algorithm
        while j < 99:
            env.render()
            j += 1
            # Choose action from Q table
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            # Get new state & reward from environment
            s1, r, d, _ = env.step(a)
            # Update Q-Table with new knowledge
            Q[s, a] = Q[s, a] + eta * (r + gma * np.max(Q[s1, :]) - Q[s, a])
            rAll += r
            s = s1
            if d == True:
                break
        rev_list.append(rAll)
        env.render()
    print ("Reward Sum on all episodes " + str(sum(rev_list) / epis))
    print("Final Values Q-Table")
    print(Q)

if __name__=="__main__":

    map_name_base_8 = "8x8-base"
    map_name_base_4 = "4x4-base"

    problem_id = 0

    run(problem_id, map=map_name_base_8)
