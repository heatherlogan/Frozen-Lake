from aima3.search import *
from uofgsocsai import LochLomondEnv
from helpers import *
from run_random import *


def run_simple_agent(problem_id, map):
    reward_hole = 0.0

    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole, map_name_base=map)
    env.render()

    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

    frozen_lake_map = UndirectedGraph(state_space_actions)
    frozen_lake_map.locations = state_space_locations

    frozen_lake_problem = GraphProblem(state_initial_id, state_goal_id, frozen_lake_map)

    # print("Initial state: " + frozen_lake_problem.initial)
    # print("Goal state: "    + frozen_lake_problem.goal)

    iterations, all_node_colors, node = my_astar_search_graph(problem=frozen_lake_problem, h=None)

    solution_path = [node]
    cnode = node.parent
    solution_path.append(cnode)
    while cnode.state != "S_00_00":
        cnode = cnode.parent
        if cnode is None:
            break
        solution_path.append(cnode)

    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)
    maze_map = UndirectedGraph(state_space_actions)
    maze_map.locations = state_space_locations
    maze_problem = GraphProblem(state_initial_id, state_goal_id, maze_map)
    iterations, _, node = my_astar_search_graph(problem=maze_problem, h=None)

    solution_path = [node]
    cnode = node.parent
    solution_path.append(cnode)
    while cnode.state != state_initial_id:
        cnode = cnode.parent
        solution_path.append(cnode)
    #
    # print("----------------------------------------")
    # print("goal state:"+str(solution_path[0]))
    # print("----------------------------------------")
    # print("Solution trace:"+str(solution_path))
    # print("----------------------------------------")

    # number of times goal is reached out of max_episodes/ (performance measures where reward is collected)
    goal_episodes = []
    # average number of iterations taken to reach goal per rewarded episode
    goal_iterations = []
    # number of episodes before goal is first reached
    first_goal = 0
    max_episodes = 10000

    for e in range(max_episodes):

        steps = solution_path[::-1]

        for step in range(len(steps)-1):

            #env.render()
            action = get_action_from_states(steps[step], steps[step+1]) # your agent goes here

            _, _, done, _ = env.step(action)

            if(done):
                print(problem_id)
                #env.render()
                goal_episodes.append(e)
                goal_iterations.append(iterations)
                if first_goal == 0:
                    first_goal = iterations
                # print("We have reached the goal in {} steps".format(iterations))
                break
    return len(goal_episodes), max_episodes-len(goal_episodes), \
           mean(goal_iterations), mini(goal_iterations), maxi(goal_iterations)

if __name__=='__main__':
    map_name_base_8 = "8x8-base"
    map_name_base_4 = "4x4-base"

    file = open('output_tables/simple_agent_8x8.csv', 'w')
    file.write("problem_id, avg rewards in 10000 eps, avg failures in 10000 eps, avg iterations to reward, avg best-case iterations to reward, avg worst case iterations to reward\n")


    for i in range(0,8):
        # print(run_simple_agent(problem_id=i, map=map_name_base_8))
        a,b,c,d,e = run_simple_agent(problem_id=i, map=map_name_base_8)
        file.write("problem_{}, {}, {},{},{},{}\n".format(i,a,b,c,d,e))