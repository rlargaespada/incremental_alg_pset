from collections import defaultdict
from itertools import tee
from typing import NamedTuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def in_notebook() -> bool:
    try:
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


GRAPH_SMALL = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 0],
])

GRAPH_LARGE = np.zeros((29, 49))
GRAPH_LARGE[14, 9:20] = 1
GRAPH_LARGE[14:28, 19] = 1
GRAPH_LARGE[1:15, 29] = 1
GRAPH_LARGE[14:29, 39] = 1


class State(NamedTuple):
    """A state in a graph represented by a 2D array."""
    x: int
    y: int


@dataclass
class ARAStar_State:
    """A snapshot of the different States being tracked by ARA*,
    as well as the current path being expanded."""
    OPEN: dict
    CLOSED: set
    INCONS: set
    current_path: list[State]


class ARAStar_Plotter:
    CELL_TYPES = ['CLEAR', 'OBSTACLE', 'OPEN', 'INCONS', 'CLOSED']
    CELL_COLORS = ['white', 'black', 'dodgerblue', 'chartreuse', 'silver']
    UNKNOWN_CELL_COLOR = 'red'
    PATH_COLOR = 'red'
    GOAL_COLOR = 'magenta'
    PATH_MARKER = 'x'
    COLORMAP = ListedColormap(CELL_COLORS, 'ARA*_planner').with_extremes(
        under=UNKNOWN_CELL_COLOR, over=UNKNOWN_CELL_COLOR)
    COLOR_BOUNDS = BoundaryNorm(np.arange(len(CELL_TYPES) + 1), COLORMAP.N)

    def __init__(self, graph: np.ndarray, start: State, goal: State):
        self.graph = graph
        self.start = start
        self.goal = goal
        self._selected_axes = {}

    def visualize_graph(self):
        fig, ax = plt.subplots()
        fig.canvas.toolbar_visible = False
        fig.canvas.footer_visible = False

        ax.pcolormesh(self.graph, cmap=self.COLORMAP, norm=self.COLOR_BOUNDS,
                      edgecolors='k', linewidth=0.5)
        ax.invert_yaxis()  # do this otherwise graph looks upside down
        ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, bottom=False)

        # plot start and goal markers
        ax.plot(self.start.y + 0.5, self.start.x + 0.5, color=self.PATH_COLOR,
                marker=self.PATH_MARKER, linestyle='none')
        ax.plot(self.goal.y + 0.5, self.goal.x + 0.5, color=self.GOAL_COLOR,
                marker=self.PATH_MARKER, linestyle='none')

        # add start/goal to legend
        handles = []
        handles.append(Line2D([], [], color=self.PATH_COLOR, linestyle='none',
                              marker=self.PATH_MARKER,label='Start'))
        handles.append(Line2D([], [], color=self.GOAL_COLOR, linestyle='none',
                              marker=self.PATH_MARKER, label='Goal'))
        ax.legend(handles=handles)

        plt.show()


    def _add_legend(self, ax: plt.Axes):
        handles = []
        for cell_type, color in zip(self.CELL_TYPES, self.CELL_COLORS):
            p = Patch(facecolor=color, label=cell_type, edgecolor='k')
            handles.append(p)

        handles.append(Line2D([], [], color=self.PATH_COLOR,
                              marker=self.PATH_MARKER,label='Path'))
        handles.append(Line2D([], [], color=self.GOAL_COLOR, linestyle='none',
                              marker=self.PATH_MARKER, label='Goal'))
        ax.legend(handles=handles)

    def _path_to_xyvals(self, path: list[State]) -> tuple[list[float], list[float]]:
        # reverse x/y order since graph is inverted
        xvals, yvals = [], []
        for x, y in path:
            xvals.append(y + 0.5)
            yvals.append(x + 0.5)
        return xvals, yvals

    def _add_path(self, ax: plt.Axes, path: list[State]):
        xvals, yvals = self._path_to_xyvals(path)
        ax.plot(xvals, yvals, color=self.PATH_COLOR)
        ax.plot(xvals[0], yvals[0], xvals[-1], yvals[-1], color=self.PATH_COLOR,
                marker=self.PATH_MARKER, linestyle='none')  # plot path start/end markers
        ax.plot(self.goal.y + 0.5, self.goal.x + 0.5, color=self.GOAL_COLOR,
                marker=self.PATH_MARKER, linestyle='none')  # plot goal marker

    def plot_episode(self, epsilon: float, history: list[ARAStar_State]):
        with plt.ioff():
            fig = plt.figure()
        fig.canvas.toolbar_visible = False
        fig.canvas.footer_visible = False
        axs = []

        for i, alg_state in enumerate(history):
            ax = fig.add_axes([0.125, 0.12, .8, 0.75], label=i, visible=i == 0)
            graph = self.graph.copy()
            for s in alg_state.OPEN:
                graph[s] = self.CELL_TYPES.index('OPEN')
            for s in alg_state.INCONS:
                graph[s] = self.CELL_TYPES.index('INCONS')
            for s in alg_state.CLOSED:
                graph[s] = self.CELL_TYPES.index('CLOSED')

            ax.pcolormesh(graph, cmap=self.COLORMAP, norm=self.COLOR_BOUNDS,
                              edgecolors='k', linewidth=0.5)
            self._add_path(ax, alg_state.current_path)

            ax.set_title(rf'Anytime Replanning A*, $\epsilon={epsilon}$, '
                         f'Iteration {i}')
            ax.invert_yaxis()  # do this otherwise graph looks upside down
            ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, bottom=False)
            self._add_legend(ax)
            axs.append(ax)

        self._selected_axes[epsilon] = 0
        def select_ax(new_ax):
            current_ax = self._selected_axes[epsilon]
            new_ax %= len(history)
            if new_ax != current_ax:
                axs[current_ax].set_visible(False)
                axs[new_ax].set_visible(True)
                self._selected_axes[epsilon] = new_ax
                fig.canvas.draw_idle()

        bforward = widgets.Button(
            disabled=False,
            button_style='',
            icon='caret-right'
        )
        bbackward = widgets.Button(
            disabled=False,
            button_style='',
            icon='caret-left'
        )
        bbackward.on_click(lambda _: select_ax(self._selected_axes[epsilon] - 1))
        bforward.on_click(lambda _: select_ax(self._selected_axes[epsilon] + 1))
        footer = widgets.HBox([bbackward, bforward])
        return widgets.VBox([fig.canvas, footer])

    def plot_final_state(self, epsilon: float, history: list[ARAStar_State]):
        fig, ax = plt.subplots()
        fig.canvas.toolbar_visible = False
        fig.canvas.footer_visible = False

        alg_state = history[-1]  # only care about final expansion
        graph = self.graph.copy()
        for s in alg_state.OPEN:
            graph[s] = self.CELL_TYPES.index('OPEN')
        for s in alg_state.INCONS:
            graph[s] = self.CELL_TYPES.index('INCONS')
        for s in alg_state.CLOSED:
            graph[s] = self.CELL_TYPES.index('CLOSED')

        ax.pcolormesh(graph, cmap=self.COLORMAP, norm=self.COLOR_BOUNDS,
                            edgecolors='k', linewidth=0.5)
        self._add_path(ax, alg_state.current_path)

        ax.set_title(rf'Anytime Replanning A*, $\epsilon={epsilon}$')
        ax.invert_yaxis()  # do this otherwise graph looks upside down
        ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, bottom=False)
        self._add_legend(ax)

        plt.show()

    def plot_paths_found(self, paths_found: dict[float, list[State]]):
        fig, ax = plt.subplots()
        fig.canvas.toolbar_visible = False
        fig.canvas.footer_visible = False

        ax.pcolormesh(self.graph, cmap=self.COLORMAP, norm=self.COLOR_BOUNDS,
                                    edgecolors='k', linewidth=0.5)
        ax.set_title('Anytime Replanning A*: All Paths Found')
        ax.invert_yaxis()  # do this otherwise graph looks upside down
        ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, bottom=False)

        for epsilon, path in paths_found.items():
            xvals, yvals = self._path_to_xyvals(path)
            ax.plot(xvals, yvals, label=rf'$\epsilon={round(epsilon, 2)}$')

        ax.legend()
        plt.show()


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

    def path_cost(self, epsilon: float) -> float:
        """Calculates the total cost for the path found using the given
        epsilon value. If no path exists for this epsilon, returns 0."""
        path = self.paths_found.get(epsilon)
        if not path:
            return 0

        cost = 0
        for s1, s2 in _pairwise(path):
            cost += self.cost(s1, s2)
        return cost

    def all_path_costs(self) -> dict[float: float]:
        """Returns path costs for all found paths."""
        return {eps: self.path_cost(eps) for eps in self.paths_found}
