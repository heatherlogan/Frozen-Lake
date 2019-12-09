from aima3.search import *
from uofgsocsai import LochLomondEnv
from helpers import *
from run_random import *


def my_best_first_graph_search_for_vis(problem, f):
    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}

    f = memoize(f, 'f')
    node = Node(problem.initial)

    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return (iterations, all_node_colors, node)

    frontier = PriorityQueue('min', f)
    frontier.append(node)

    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    explored = set()
    while frontier:
        node = frontier.pop()

        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None

def heuristic(problem, node):

    def getxy(strnode):
        s, x, y = strnode.split('_')
        return int(x), int(y)
    x1, y1 = getxy(node.state)
    x2, y2 = getxy(problem.goal)
    # return abs(x2 - x1) + abs(y2 - y1) #manhattan
    # return np.sqrt((x2-x1)**2 + (y2-y1)**2) #euclidian
    return 0


def my_astar_search_graph(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    # h = memoize(h or problem.h, 'h')
    iterations, all_node_colors, node = my_best_first_graph_search_for_vis(problem,
                                                                lambda n: n.path_cost + heuristic(problem, n))
    return(iterations, all_node_colors, node)


# Very basic implementation of a node state to action parser

def get_action_from_states(cur_node, next_node):
    # Action to int representations (taken from the FrozenLake github page)
    # (https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    # Get the coordinates from the state string for each node
    x1 = cur_node.state[2]
    y1 = cur_node.state[4]
    x2 = next_node.state[2]
    y2 = next_node.state[4]

    # We need to account for rotation between graph and environment
    # X on our graph became Y on our environment (handles up/down)
    # Y on our graph became X on our environment (handles left/right)
    if x1 == x2:
        if y1 > y2:
            return LEFT
        else:
            return RIGHT
    else:
        if x1 > x2:
            return UP
        else:
            return DOWN



def run_simple_agent(problem_id, map):

    reward_hole = 0.00

    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole, map_name_base=map)
    env.reset()
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

    frozen_lake_map = UndirectedGraph(state_space_actions)
    frozen_lake_map.locations = state_space_locations

    frozen_lake_problem = GraphProblem(state_initial_id, state_goal_id, frozen_lake_map)

    iterations, all_node_colors, node = my_astar_search_graph(problem=frozen_lake_problem, h=heuristic)

    solution_path = [node]
    cnode = node.parent
    solution_path.append(cnode)

    while cnode.state != state_initial_id:
        cnode = cnode.parent
        solution_path.append(cnode)

    # number of times goal is reached out of max_episodes/ (performance measures where reward is collected)
    goal_episodes = []
    # average number of iterations taken to reach goal per rewarded episode
    goal_iterations = []
    # number of episodes before goal is first reached
    first_goal = 0

    rewards = []

    max_episodes = 10000

    steps = solution_path[::-1]


    for e in range(max_episodes):

        state = env.reset()

        rewards_current_episode = 0

        for step in range(len(steps)):

            # env.render()

            action = get_action_from_states(steps[step], steps[step+1])

            state, reward, done, info = env.step(action)

            rewards_current_episode += 1.0

            if(done):
                goal_episodes.append(e)

                # + 2 for start and goal states
                final_steps = step + 2
                goal_iterations.append(final_steps)

                if first_goal == 0:
                    first_goal = e
                break

        rewards.append(rewards_current_episode)

    rewards_per_100_eps = np.split(np.array(rewards), max_episodes / 100)
    rewards_per_100_eps = [str(sum(r / 100)) for r in rewards_per_100_eps]

    return len(goal_episodes), max_episodes-len(goal_episodes), \
           mean(goal_iterations), mini(goal_iterations), maxi(goal_iterations), first_goal, rewards_per_100_eps

if __name__=='__main__':

    problem_id = sys.argv[1]
    map_name = sys.argv[2]

    results = run_simple_agent(int(problem_id), map_name)

    idx= ['Total Rewards in 10000 eps', 'Total Failures in 10000 eps', 'Avg Steps to Goal', 'Best-case steps to Goal', 'Worst-case steps to Goal', 'Episodes to first Goal']
    df = pd.DataFrame((results[:-1]), columns=['problem_{}'.format(problem_id)])

    df['Measure'] = idx
    df = df.set_index('Measure')
    print('---------------SIMPLE AGENT---------------\n', df)


