import numpy as np

from IPython.display import display_html
from nose.tools import assert_equal, assert_almost_equal

from utils import State, ARAStar_Planner, GRAPH_LARGE, GRAPH_SMALL, in_notebook


def test_ok():
    """If execution gets to this point, print out a happy message."""
    if in_notebook():
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    else:
        print("Tests passed!!")


def test_initalize(fn):
    start = State(0, 0)
    goal = State(6, 5)
    planner = ARAStar_Planner(GRAPH_SMALL, start, goal, 2.5, stepsize=1)
    fn(planner)

    assert_equal(planner.g, {goal: np.inf, start: 0})
    assert_almost_equal(planner.OPEN[start], 19.525624189766635)
    assert_equal(planner.CLOSED, set())
    assert_equal(planner.INCONS, set())
    assert_equal(planner.PARENTS, {start: start})
    assert_equal(planner.alg_history, dict())
    assert_equal(planner.paths_found, dict())

    start = State(24, 4)
    goal = State(4, 44)
    planner = ARAStar_Planner(GRAPH_LARGE, start, goal, 1.5, .2)
    fn(planner)
    assert_almost_equal(planner.OPEN[start], 67.0820393249937)

    test_ok()


def _test_alg_hist(open_user: dict, open_soln: dict, closed_user: set, closed_soln: set):
    for s, fval in open_soln.items():
        assert_almost_equal(fval, open_user.get(s, -1))
    assert_equal(closed_user, closed_soln)


def test_improve_path(fn, initialize):
    start = State(0, 0)
    goal = State(6, 5)
    eps = 2.5
    planner = ARAStar_Planner(GRAPH_SMALL, start, goal, eps, stepsize=1)
    initialize(planner)
    fn(planner)
    planner.publish_path()

    path = {eps: [
        State(x=0, y=0),
       State(x=1, y=1),
       State(x=2, y=1),
       State(x=3, y=1),
       State(x=4, y=0),
       State(x=5, y=1),
       State(x=5, y=2),
       State(x=4, y=3),
       State(x=3, y=4),
       State(x=4, y=5),
       State(x=5, y=5),
       State(x=6, y=5)]}
    assert_equal(planner.paths_found, path)

    user_hist = planner.alg_history[eps][1]
    OPEN_soln = {
        State(x=0, y=1): 19.027756377319946,
        State(x=1, y=1): 17.422024155955217,
    }
    CLOSED_soln = {start}
    _test_alg_hist(user_hist.OPEN, OPEN_soln, user_hist.CLOSED, CLOSED_soln)

    user_hist = planner.alg_history[eps][2]
    OPEN_soln = {
        State(x=0, y=1): 19.027756377319946,
        State(x=0, y=2): 19.598936955994617,
        State(x=2, y=1): 16.556349186104047
    }
    CLOSED_soln = {start, State(x=1, y=1)}
    _test_alg_hist(user_hist.OPEN, OPEN_soln, user_hist.CLOSED, CLOSED_soln)

    user_hist = planner.alg_history[eps][3]
    OPEN_soln = {
        State(x=0, y=1): 19.027756377319946,
        State(x=0, y=2): 19.598936955994617,
        State(x=3, y=0): 18.40580686185944,
        State(x=3, y=1): 15.914213562373096
    }
    CLOSED_soln = {start, State(x=1, y=1), State(x=2, y=1)}
    _test_alg_hist(user_hist.OPEN, OPEN_soln, user_hist.CLOSED, CLOSED_soln)

    user_hist = planner.alg_history[eps][-1]
    OPEN_soln = {
        State(x=0, y=1): 19.027756377319946,
        State(x=0, y=2): 19.598936955994617,
        State(x=2, y=5): 21.48528137423857,
        State(x=3, y=0): 18.40580686185944,
        State(x=3, y=3): 18.670732438152353,
        State(x=3, y=5): 18.571067811865476,
        State(x=5, y=0): 18.575975908728154,
        State(x=6, y=5): 13.485281374238571
    }
    CLOSED_soln = {
        State(x=0, y=0),
        State(x=1, y=1),
        State(x=2, y=1),
        State(x=3, y=1),
        State(x=3, y=4),
        State(x=4, y=0),
        State(x=4, y=3),
        State(x=4, y=5),
        State(x=5, y=1),
        State(x=5, y=2),
        State(x=5, y=3),
        State(x=5, y=5)
    }
    _test_alg_hist(user_hist.OPEN, OPEN_soln, user_hist.CLOSED, CLOSED_soln)

    test_ok()


def test_calc_epsilon_prime(fn, improve_path, initialize):
    planner = ARAStar_Planner(GRAPH_SMALL, State(0, 0), State(6, 5), 2.5, stepsize=1)
    initialize(planner)
    improve_path(planner)
    assert_almost_equal(fn(planner), 1.6423228537944068)

    planner = ARAStar_Planner(GRAPH_LARGE, State(24, 4), State(4, 44), 1.5, .2)
    initialize(planner)
    improve_path(planner)
    assert_almost_equal(fn(planner), 1.2514705177523118)

    test_ok()


def test_run(fn):
    planner = ARAStar_Planner(GRAPH_LARGE, State(24, 4), State(4, 44), 1.5, .2)
    fn(planner)

    paths_found_soln = {
        1: [State(x=24, y=4),
            State(x=24, y=5),
            State(x=24, y=6),
            State(x=24, y=7),
            State(x=25, y=8),
            State(x=26, y=9),
            State(x=27, y=10),
            State(x=27, y=11),
            State(x=27, y=12),
            State(x=27, y=13),
            State(x=27, y=14),
            State(x=27, y=15),
            State(x=27, y=16),
            State(x=27, y=17),
            State(x=27, y=18),
            State(x=28, y=19),
            State(x=27, y=20),
            State(x=26, y=21),
            State(x=25, y=22),
            State(x=24, y=23),
            State(x=23, y=24),
            State(x=22, y=25),
            State(x=21, y=26),
            State(x=20, y=27),
            State(x=19, y=28),
            State(x=18, y=29),
            State(x=17, y=30),
            State(x=16, y=31),
            State(x=15, y=32),
            State(x=14, y=33),
            State(x=13, y=34),
            State(x=12, y=35),
            State(x=11, y=36),
            State(x=10, y=37),
            State(x=9, y=38),
            State(x=8, y=39),
            State(x=7, y=40),
            State(x=6, y=41),
            State(x=5, y=42),
            State(x=4, y=43),
            State(x=4, y=44)],
        1.1: [State(x=24, y=4),
            State(x=23, y=5),
            State(x=22, y=5),
            State(x=21, y=6),
            State(x=20, y=7),
            State(x=19, y=8),
            State(x=18, y=8),
            State(x=17, y=8),
            State(x=16, y=8),
            State(x=15, y=8),
            State(x=14, y=8),
            State(x=13, y=9),
            State(x=13, y=10),
            State(x=13, y=11),
            State(x=13, y=12),
            State(x=13, y=13),
            State(x=13, y=14),
            State(x=13, y=15),
            State(x=13, y=16),
            State(x=13, y=17),
            State(x=13, y=18),
            State(x=13, y=19),
            State(x=13, y=20),
            State(x=13, y=21),
            State(x=13, y=22),
            State(x=13, y=23),
            State(x=13, y=24),
            State(x=13, y=25),
            State(x=13, y=26),
            State(x=13, y=27),
            State(x=14, y=28),
            State(x=15, y=29),
            State(x=14, y=30),
            State(x=13, y=31),
            State(x=12, y=32),
            State(x=11, y=33),
            State(x=10, y=34),
            State(x=9, y=35),
            State(x=8, y=36),
            State(x=7, y=37),
            State(x=6, y=38),
            State(x=6, y=39),
            State(x=6, y=40),
            State(x=5, y=41),
            State(x=5, y=42),
            State(x=4, y=43),
            State(x=4, y=44)],
        1.3: [State(x=24, y=4),
            State(x=23, y=5),
            State(x=22, y=6),
            State(x=21, y=7),
            State(x=20, y=8),
            State(x=19, y=8),
            State(x=18, y=8),
            State(x=17, y=8),
            State(x=16, y=8),
            State(x=15, y=8),
            State(x=14, y=8),
            State(x=13, y=9),
            State(x=13, y=10),
            State(x=13, y=11),
            State(x=13, y=12),
            State(x=13, y=13),
            State(x=13, y=14),
            State(x=12, y=15),
            State(x=12, y=16),
            State(x=12, y=17),
            State(x=11, y=18),
            State(x=10, y=19),
            State(x=9, y=20),
            State(x=8, y=21),
            State(x=7, y=22),
            State(x=6, y=23),
            State(x=5, y=24),
            State(x=4, y=25),
            State(x=3, y=26),
            State(x=2, y=27),
            State(x=1, y=28),
            State(x=0, y=29),
            State(x=0, y=30),
            State(x=0, y=31),
            State(x=1, y=32),
            State(x=1, y=33),
            State(x=1, y=34),
            State(x=1, y=35),
            State(x=2, y=36),
            State(x=2, y=37),
            State(x=2, y=38),
            State(x=3, y=39),
            State(x=3, y=40),
            State(x=3, y=41),
            State(x=3, y=42),
            State(x=4, y=43),
            State(x=4, y=44)],
        1.5: [State(x=24, y=4),
            State(x=23, y=5),
            State(x=22, y=6),
            State(x=21, y=7),
            State(x=20, y=8),
            State(x=19, y=8),
            State(x=18, y=8),
            State(x=17, y=8),
            State(x=16, y=8),
            State(x=15, y=8),
            State(x=14, y=8),
            State(x=13, y=9),
            State(x=13, y=10),
            State(x=13, y=11),
            State(x=13, y=12),
            State(x=13, y=13),
            State(x=13, y=14),
            State(x=12, y=15),
            State(x=12, y=16),
            State(x=12, y=17),
            State(x=11, y=18),
            State(x=10, y=19),
            State(x=9, y=20),
            State(x=8, y=21),
            State(x=7, y=22),
            State(x=6, y=23),
            State(x=5, y=24),
            State(x=4, y=25),
            State(x=3, y=26),
            State(x=2, y=27),
            State(x=1, y=28),
            State(x=0, y=29),
            State(x=0, y=30),
            State(x=0, y=31),
            State(x=1, y=32),
            State(x=1, y=33),
            State(x=1, y=34),
            State(x=1, y=35),
            State(x=2, y=36),
            State(x=2, y=37),
            State(x=2, y=38),
            State(x=3, y=39),
            State(x=3, y=40),
            State(x=3, y=41),
            State(x=3, y=42),
            State(x=4, y=43),
            State(x=4, y=44)]}

    for eps, path in paths_found_soln.items():
        assert_equal(path, planner.paths_found.get(eps))

    test_ok()
