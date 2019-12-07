import sys
from run_random import *
from run_simple import *
from run_random import *
from run_rl import *

if __name__=='__main__':

    results_random = []
    results_simple = []
    results_ri = []

    map_name = sys.argv[1]

    if map_name not in ["8x8-base", "4x4-base"]:
        print('Invalid Map, using 4x4-base map')
        map_name = '4x4-base'

    if map_name=='4x4-base':
        p_id_range = range(0,3)
    else:
        p_id_range = range(0,8)

    # run scripts over every problem id

    for i in p_id_range:

        # run random_agent
        random_r = run_senseless_agent(problem_id=i, map=map_name)
        results_random.append(random_r)


        # run simple agent
        simple_r = run_simple_agent(problem_id=i, map=map_name)
        results_simple.append(simple_r)

        # run reinforcement agent
        ri_r = run_reinforcement_agent(problem_id=i, map=map_name)
        results_random.append(ri_r)


    # format results


    # take averages of each metric for each



    # make figures



    # learning behaviour and convergence of agents


    pass