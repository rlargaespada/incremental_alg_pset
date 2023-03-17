from collections import defaultdict
import numpy as np

from utils import State, ARAStar_State

class ARAStar_Planner:
    """The ARA* planner with needed utilities, along with some additional
    utilities for serializing the state of the algorithm for visualization
    and testing purposes.

    Attributes
    ----------
    graph : np.ndarray, (n x m)
        the graph being searched through, see __init__ for format requirements.
    start, goal : State
        start and goal states in the graph
    epsilon : float
        inflation factor for the heuristic function
    stepsize : float
        amount to decrease epsilon by each iteration of ARA*
    g : dict[State, float]
        dict of costs from the start State to a given State within the graph
    OPEN : dict[State: float]
        dict of locally inconsistent States to respective fvalues, used as
        a priority queue for ARA* search
    CLOSED : set[State]
        set of States which have been expanded
    INCONS : set[State]
        set of locally inconsistent States which have been previously expanded
    PARENTS : dict[State, State]
        dict mapping each expanded State to its predecessor State in the graph
    alg_history : dict[float, List[ARAStar_State]]
        dict mapping an epsilon value to a list of ARAStar_State objects,
        which recreate the state of the algoritm as it progressed through
        its search.
    paths_found : dict[float, List[State]]
        dict mapping epsilon values to the final paths returned for that
        epsilon. Path is represented as a list of States, start to goal.
    """
    def __init__(self, graph: np.ndarray, start: State,
                 goal: State, epsilon: float = 3.0,
                 stepsize: float = 0.4) -> None:
        '''
        Parameters
        ----------
            graph : np.ndarray
                NumPy array representing the graph to search through. Should
                comprise of only 0s and 1s, with 0s representing free space
                and 1s representing obstacles.
            start, goal : State
                start and goal states for the search.
            epsilon : float
                initial inflation factor for the heuristic function
            stepsize : float
                amount to decrease epsilon by each iteration of ARA*
        '''
        self.graph = graph
        self.start = start
        self.goal = goal
        self.epsilon = epsilon
        self.stepsize = stepsize

        self.g = {}
        self.OPEN = {}
        self.CLOSED = set()
        self.INCONS = set()
        self.PARENTS = {}

        self.alg_history = defaultdict(list)
        self.paths_found = {}

    def h(self, state: State) -> int:
        """Euclidean heuristic between goal and state."""
        return np.hypot(self.goal.x - state.x, self.goal.y - state.y)
        # return max(abs(state.x - self.goal.x), abs(state.y - self.goal.y))

    def f(self, state: State) -> float:
        """Combined inflated heuristic."""
        return self.g[state] + self.epsilon * self.h(state)

    def is_clear(self, state: State) -> bool:
        """Returns True if given state does not collide with an obstacle in the graph,
        False otherwise."""
        return self.graph[state] == 0

    def is_obstacle(self, state: State):
        """Returns True if given state collides with an obstacle in the graph,
        False otherwise."""
        return self.graph[state] != 0

    def valid_state(self, state: State) -> bool:
        """Returns True if given state is within bounds of graph and does not
        collide with an obstacle, False otherwise."""
        x, y = state
        x_bound, y_bound = self.graph.shape
        return 0 <= x < x_bound and 0 <= y < y_bound and self.is_clear(state)

    def neighbors(self, state: State) -> list[State]:
        """Returns list of neighbors of the given state which are within the bounds
        of the 8-connected graph and which do not collide with obstacles."""
        x, y = state
        n = [
            (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
            (x, y - 1), (x, y + 1),
            (x + 1, y - 1), (x + 1, y), (x + 1, y + 1),
        ]
        return [State(*s) for s in n if self.valid_state(s)]

    def cost(self, state1: State, state2: State) -> float:
        """Cost of traversal between two states. Infinite if states are not
        neighbors, else Euclidean distance."""
        if state2 not in self.neighbors(state1):
            return np.inf

        return np.hypot(state2.x - state1.x, state2.y - state1.y)

    def get_next_state(self) -> State:
        """Returns the state from OPEN with the lowest f value, which should
        be expanded next, or None if OPEN is empty."""
        if not self.OPEN:
            return None

        lowest_f = min(self.OPEN.values())
        next_states = {state for state, f in self.OPEN.items() if f == lowest_f}
        return min(next_states)  # breaks ties between states

    def extract_path(self, final_state: State = None) -> list[State]:
        """From PARENTS mapping, returns path to final_state as a list
        of States. If final_state is None, defaults to goal state."""
        if final_state is None:
            final_state = self.goal

        if final_state not in self.PARENTS:  # path not found yet
            return None

        s = final_state  # rename within loop
        path = [s]
        while True:
            s = self.PARENTS[s]
            path.append(s)

            if s == self.start:
                break

        path.reverse()
        return path

    def publish_path(self):
        """Saves current value of epsilon and current path to set of paths found."""
        self.paths_found[self.epsilon] = self.extract_path()

    def save_alg_state(self, current_state: State):
        """Saves current path and values of OPEN, CLOSED, and INCONS states
        to alg history for use in testing and plotting."""
        self.alg_history[self.epsilon].append(ARAStar_State(
                self.OPEN.copy(),
                self.CLOSED.copy(),
                self.INCONS.copy(),
                self.extract_path(current_state)
            ))


#* student implemented
def initialize(planner: ARAStar_Planner):
    planner.g[planner.goal] = np.inf
    planner.g[planner.start] = 0
    planner.OPEN.clear()
    planner.CLOSED.clear()
    planner.INCONS.clear()
    planner.PARENTS.clear()  #^ new
    planner.OPEN[planner.start] = planner.f(planner.start)
    planner.PARENTS[planner.start] = planner.start  #^ new
    planner.alg_history.clear()  #^ new
    planner.paths_found.clear()  #^ new


#* student implemented
def improve_path(planner: ARAStar_Planner):
    planner.save_alg_state(planner.start)  #^ new
    while planner.f(planner.goal) > min(planner.f(s) for s in planner.OPEN):
        s = planner.get_next_state()
        planner.OPEN.pop(s)
        planner.CLOSED.add(s)

        for sprime in planner.neighbors(s):
            if sprime not in planner.g:  # unvisited
                planner.g[sprime] = np.inf

            new_cost = planner.g[s] + planner.cost(s, sprime)
            if planner.g[sprime] > new_cost:
                planner.g[sprime] = new_cost
                planner.PARENTS[sprime] = s  #^ new

                if sprime not in planner.CLOSED:
                    planner.OPEN[sprime]  = planner.f(sprime)
                else:
                    planner.INCONS.add(sprime)
        planner.save_alg_state(s)  #^ new

#* student implemented
def calc_epsilon_prime(planner: ARAStar_Planner):
    open_plus_incons = list(planner.OPEN.keys()) + list(planner.INCONS)
    if open_plus_incons:
        v = min(planner.g[s] + planner.h(s) for s in open_plus_incons)
    else:
        v = np.inf

    return min(planner.epsilon, planner.g[planner.goal] / v)


#* student implemented
def run(planner: ARAStar_Planner):
    initialize(planner)  #^ new, optional
    improve_path(planner)
    epsilon_prime = calc_epsilon_prime(planner)  #^ new, optional
    planner.publish_path()

    while epsilon_prime > 1:
        planner.epsilon = max(1, planner.epsilon - planner.stepsize)

        while planner.INCONS:
            planner.OPEN[planner.INCONS.pop()] = 0
        planner.OPEN = {s: planner.f(s) for s in planner.OPEN}

        planner.CLOSED.clear()
        improve_path(planner)
        epsilon_prime = calc_epsilon_prime(planner)  #^ new, optional
        planner.publish_path()


if __name__ == '__main__':
    import utils
    test = ARAStar_Planner(utils.GRAPH_LARGE, State(24, 4), State(4, 44), 1.5, .2)
    plotter = utils.ARAStar_Plotter(utils.GRAPH_LARGE, State(24, 4), State(4, 44))
    # test = ARAStar_Planner(utils.GRAPH_SMALL, State(0, 0), State(6, 5), 2.5, 1)
    # plotter = utils.ARAStar_Plotter(utils.GRAPH_SMALL, State(0, 0), State(6, 5))

    run(test)

    # plotter.visualize_graph()
    # plotter.plot_episode(eps, test.alg_history[eps])
    # plotter.plot_final_state(eps, test.alg_history[eps])
    plotter.plot_paths_found(test.paths_found)


    # EPSILON = 2.5
    # START = State(0, 0)
    # GOAL = State(6, 5)

    # planner = ARAStar_Planner(utils.GRAPH_SMALL, START, GOAL, EPSILON, stepsize=1)
    # plotter = utils.ARAStar_Plotter(utils.GRAPH_SMALL, START, GOAL)
    # run(planner)
    # plotter.plot_paths_found(planner.paths_found)
