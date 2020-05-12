# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns the goal state for the search problem. This is required for bi-directional search.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    trace = util.Stack()
    starting_state = problem.getStartState()
    trace.push((starting_state, list()))
    # empty list will eventually hold the actions taken to reach current element

    visited_states = set()

    while not trace.isEmpty():
        curr_state, actions = trace.pop()
        if curr_state not in visited_states:
            if problem.isGoalState(curr_state):
                return actions
            successors = problem.getSuccessors(curr_state)
            for successor in successors:
                trace.push((successor[0], actions[:] + [successor[1]]))
            visited_states.add(curr_state)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    trace = util.Queue()
    starting_state = problem.getStartState()
    trace.push((starting_state, list()))
    # empty list will eventually hold the actions taken to reach current element

    visited_states = set()

    while not trace.isEmpty():
        curr_state, actions = trace.pop()
        if curr_state not in visited_states:
            if problem.isGoalState(curr_state):
                return actions
            successors = problem.getSuccessors(curr_state)
            for successor in successors:
                trace.push((successor[0], actions[:] + [successor[1]]))
            visited_states.add(curr_state)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    trace = util.PriorityQueue()
    starting_state = problem.getStartState()
    cost = 0
    trace.push((starting_state, list(), cost), cost)
    # empty list will eventually hold the actions taken to reach current element
    # placeholder will carry on the costs

    visited_states = set()

    while not trace.isEmpty():
        curr_state, actions, cost = trace.pop()
        if curr_state not in visited_states:
            if problem.isGoalState(curr_state):
                return actions
            successors = problem.getSuccessors(curr_state)
            for successor in successors:
                trace.push((successor[0], actions[:] + [successor[1]], cost + successor[2]), cost + successor[2])
            visited_states.add(curr_state)


def nullHeuristic(state, problem=None, info={}):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    trace = util.PriorityQueue()
    starting_state = problem.getStartState()
    cost = 0
    trace.push((starting_state, list(), cost), cost)
    # empty list will eventually hold the actions taken to reach current element
    # placeholder will carry on the costs

    visited_states = set()

    while not trace.isEmpty():
        curr_state, actions, cost = trace.pop()
        if curr_state not in visited_states:
            if problem.isGoalState(curr_state):
                # print "Path:: {}".format(actions)
                return actions
            successors = problem.getSuccessors(curr_state)
            for successor in successors:
                trace.push((successor[0], actions[:] + [successor[1]], cost + successor[2]),
                           heuristic(successor[0], problem) + cost + successor[2])
            visited_states.add(curr_state)


class BiDirectionalNode:
    def __init__(self, position, actions, cost=0, h=0, pr=0):
        self.pos = position
        self.actions = actions
        self.cost = cost
        self.f = self.cost + h
        self.priority = pr


def mirroredActions(actions):
    """
    Takes the path of actions from backward search, flips it and mirrors the directions to make sense
    from the middle point (where the two frontiers meet) onwards

    :param actions: List of actions to reach to the middle point from goal
    :return: list of actions to reach to the goal from the middle point
    """
    mirrored_actions = list(reversed(actions))      # reverse the actions
    reverse_directions = {'East': 'West', 'South': 'North', 'North': 'South', 'West': 'East'}
    mirrored_actions = [reverse_directions[action] for action in mirrored_actions]  # reverse the directions

    return mirrored_actions


def bidirectional_search(problem, heuristic=nullHeuristic):
    """Implements a front to end bidirectional heuristic search that is guaranteed to meet in middle
    from the paper http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12320/12109"""
    if heuristic is nullHeuristic:
        print("You're now in Bi-directional brute-force search")
    else:
        print("You're now in Bi-directional heuristic search")
    
    # a dictionary to keep track of flags required by bidirectional search
    reverse_heuristic = {'rev': False}

    open_f = util.PriorityQueue()   # open list for forward search
    hash_f = dict()                 # hashmap to check and remove entries from the open list for forward search 
    closed_f = dict()               # closed list for forward search

    open_b = util.PriorityQueue()   # open list for backward search
    hash_b = dict()                 # hashmap to check and remove entries from the open list for backward search 
    closed_b = dict()               # closed list for backward search

    g_f = util.PriorityQueue()      # g-value (cost) heap for forward pass
    g_b = util.PriorityQueue()      # g-value (cost) heap for backward pass
    f_f = util.PriorityQueue()      # f-value (cost + heuristic) heap for forward pass
    f_b = util.PriorityQueue()      # f-value (cost + heuristic) heap for forward pass

    U = float('inf')                # Variable U keeps track of the lowest cost path found from Bidirectional search

    start = problem.getStartState()     # gets the Starting state just like every other search algorithm
    # defined for each problem type to acquire the goal state to begin searching from backwards
    goal = problem.getGoalState()

    # note that the heuristic function takes this metadata as an optional third variable unlike in A* search
    reverse_heuristic['rev'] = False    # calculate forward heuristic
    start_node = BiDirectionalNode(start, [], 0, heuristic(start, problem, reverse_heuristic), 0)
    reverse_heuristic['rev'] = True     # calculate heuristic from goal node to start node (i.e. reverse direction)
    goal_node = BiDirectionalNode(goal, [], 0, heuristic(goal, problem, reverse_heuristic), 0)

    # initialize the forward open list, g and f heaps, closed list, and hash maps with start state
    open_f.push(start_node, start_node.priority)
    hash_f[start] = start_node
    f_f.push(start_node, start_node.f)      # prioritize by f-value
    g_f.push(start_node, start_node.cost)   # prioritize by g-value

    # initialize the backward open list, g and f heaps, closed list and hash maps with goal state
    open_b.push(goal_node, goal_node.priority)
    hash_b[goal] = goal_node
    f_b.push(goal_node, goal_node.f)
    g_b.push(goal_node, goal_node.cost)

    # all edges have unit cost, hence the minimum edge weight is hard-coded to be 1 for this case
    eps = 1

    result_action_list = []     # stores the complete path of actions required to reach from start to goal
    last_f = None               # stores the last explored node from forward search
    last_b = None               # stores the last explored node from backward search

    # the bidirectional search begins
    while not open_f.isEmpty() and not open_b.isEmpty():
        # find the node with minimum priority from both forward and backward open lists
        C = min(open_f.peek().priority, open_b.peek().priority)

        # if U <= lowerbound, return the path found so far
        if U <= max(C, f_f.peek().f, f_b.peek().f, g_f.peek().cost + g_b.peek().cost + eps):
            open_nodes = len(hash_f) + len(hash_b) + 1
            closed_nodes = len(closed_f) + len(closed_b)

            print "Total nodes generated: {}".format(open_nodes + closed_nodes)
            print "Path Length: {}".format(U)

            return result_action_list

        # safe to stop
        elif U <= C:
            return result_action_list

        # begin forward search
        if C == open_f.peek().priority:
            curr_node = last_f = open_f.pop()
            position = curr_node.pos

            # move the element from open list to closed list
            if position in hash_f:
                hash_f.pop(position)
            closed_f[position] = curr_node

            f_f.remove(curr_node)
            g_f.remove(curr_node)

            # get successors
            children = problem.getSuccessors(position)
            for c in children:
                c_state, c_action, c_cost = c
                this_node = None
                if c_state in hash_f:
                    this_node = hash_f[c_state]
                elif c_state in closed_f:
                    this_node = closed_f[c_state]

                # if node already exists
                if this_node is not None:
                    # and is reached by a shorter path then ignore cuurent path
                    if this_node.cost <= curr_node.cost + c_cost:
                        continue

                    # remove node from Open U Close Lists
                    open_f.remove(this_node)
                    if c_state in hash_f:
                        hash_f.pop(c_state)
                    if c_state in closed_f:
                        closed_f.pop(c_state)

                    f_f.remove(this_node)
                    g_f.remove(this_node)

                    # else update the cost
                    this_node.cost = (curr_node.cost + c_cost)
                    this_node.f = this_node.cost + heuristic(this_node.pos, problem)
                    this_node.actions = curr_node.actions[:] + [c_action]

                else:
                    this_actions = curr_node.actions[:] + [c_action]
                    this_cost = curr_node.cost + c_cost
                    this_heuristic = heuristic(c_state, problem)

                    this_priority = this_cost + max(this_heuristic, this_cost)

                    this_node = BiDirectionalNode(c_state, this_actions, this_cost, this_heuristic, this_priority)

                open_f.push(this_node, this_node.priority)
                hash_f[c_state] = this_node
                f_f.push(this_node, this_node.f)
                g_f.push(this_node, this_node.cost)

                # if the node is already reached from backward search, construct the path from start to goal
                if c_state in hash_b:
                    back_node = hash_b[c_state]
                    total_cost = this_node.cost + back_node.cost
                    U = min(U, total_cost)  # update minimum cost

                    backwards_actions = mirroredActions(back_node.actions)
                    result_action_list = this_node.actions[:] + backwards_actions

                    print "Path found:: Forward: {} + Backward: {} = Total cost: {}"\
                        .format(this_node.cost, back_node.cost, total_cost)
                    # print "Path:: {}".format(result_action_list)
        # end forward search

        # begin backward search
        else:
            curr_node = last_b = open_b.pop()
            position = curr_node.pos

            if position in hash_b:
                hash_b.pop(position)
            closed_b[position] = curr_node

            f_b.remove(curr_node)
            g_b.remove(curr_node)

            children = problem.getSuccessorsBS(position)
            for c in children:
                c_state, c_action, c_cost = c
                this_node = None
                if c_state in hash_b:
                    this_node = hash_b[c_state]
                elif c_state in closed_b:
                    this_node = closed_b[c_state]

                if this_node is not None:
                    if this_node.cost <= curr_node.cost + c_cost:
                        continue

                    open_b.remove(this_node)
                    if c_state in hash_b:
                        hash_b.pop(c_state)
                    if c_state in closed_b:
                        closed_b.pop(c_state)

                    f_b.remove(this_node)
                    g_b.remove(this_node)

                    reverse_heuristic['rev'] = True
                    this_node.cost = curr_node.cost + c_cost
                    this_node.f = this_node.cost + heuristic(this_node.pos, problem, reverse_heuristic)
                    this_node.actions = curr_node.actions[:] + [c_action]

                else:
                    this_actions = curr_node.actions[:] + [c_action]
                    this_cost = curr_node.cost + c_cost
                    reverse_heuristic['rev'] = True
                    this_heuristic = heuristic(c_state, problem, reverse_heuristic)

                    this_priority = this_cost + max(this_heuristic, this_cost)

                    this_node = BiDirectionalNode(c_state, this_actions, this_cost, this_heuristic, this_priority)

                open_b.push(this_node, this_node.priority)
                hash_b[c_state] = this_node
                f_b.push(this_node, this_node.f)
                g_b.push(this_node, this_node.cost)

                if c_state in hash_f:
                    front_node = hash_f[c_state]
                    total_cost = this_node.cost + front_node.cost
                    U = min(U, total_cost)

                    backwards_actions = mirroredActions(this_node.actions)
                    result_action_list = front_node.actions[:] + backwards_actions

                    print "Path found:: Backward: {} + Forward: {} = Total cost: {}"\
                        .format(this_node.cost, front_node.cost, total_cost)
                    # print "Path:: {}".format(result_action_list)
        # end backward search

    print "Both queues are empty"
    if last_f is not None and last_b is not None:
        backwards_actions = mirroredActions(last_b.actions)

        return last_f.actions + backwards_actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bihs = bidirectional_search

