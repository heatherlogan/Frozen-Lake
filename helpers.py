from aima3 import *
from aima3.search import Node
from aima3.utils import memoize, PriorityQueue


def env2statespace(env):
    """
    This simple parser demonstrates how you can extract the state space from the Open AI env

    We *assume* full observability, i.e., we can directly ignore Hole states. Alternatively,
    we could place a very high step cost to a Hole state or use a directed representation
    (i.e., you can go to a Hole state but never return). Feel free to experiment with both if time permits.

    Input:
        env: an Open AI Env follwing the std in the FrozenLake-v0 env

    Output:
        state_space_locations : a dict with the available states
        state_space_actions   : a dict of dict with available actions in each state
        state_start_id        : the start state
        state_goal_id         : the goal state

        These objects are enough to define a Graph problem using the AIMA toolbox, e.g., using
        UndirectedGraph, GraphProblem and astar_search (as in AI (H) Lab 3)

    Notice: the implementation is very explicit to demonstarte all the steps (it could be made more elegant!)

    """
    state_space_locations = {}  # create a dict
    for i in range(env.desc.shape[0]):
        for j in range(env.desc.shape[1]):
            if not (b'H' in env.desc[i, j]):
                state_id = "S_" + str(int(i)) + "_" + str(int(j))
                state_space_locations[state_id] = (int(i), int(j))
                if env.desc[i, j] == b'S':
                    state_initial_id = state_id
                elif env.desc[i, j] == b'G':
                    state_goal_id = state_id

                    # -- Generate state / action list --#
                # First define the set of actions in the defined coordinate system
                actions = {"west": [-1, 0], "east": [+1, 0], "north": [0, +1], "south": [0, -1]}
                state_space_actions = {}
                for state_id in state_space_locations:
                    possible_states = {}
                    for action in actions:
                        # -- Check if a specific action is possible --#
                        delta = actions.get(action)
                        state_loc = state_space_locations.get(state_id)
                        state_loc_post_action = [state_loc[0] + delta[0], state_loc[1] + delta[1]]

                        # -- Check if the new possible state is in the state_space, i.e., is accessible --#
                        state_id_post_action = "S_" + str(state_loc_post_action[0]) + "_" + str(
                            state_loc_post_action[1])
                        if state_space_locations.get(state_id_post_action) != None:
                            possible_states[state_id_post_action] = 1

                            # -- Add the possible actions for this state to the global dict --#
                    state_space_actions[state_id] = possible_states

    return state_space_locations, state_space_actions, state_initial_id, state_goal_id


def my_best_first_graph_search_for_vis(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

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


def my_astar_search_graph(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    iterations, all_node_colors, node = my_best_first_graph_search_for_vis(problem,
                                                                lambda n: n.path_cost + h(n))
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