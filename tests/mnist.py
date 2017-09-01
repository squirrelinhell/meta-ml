
import timer
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import Mnist, Distribution
from agents import Repeat

def print_traj(t):
    for o, a, r in t:
        print(str(o.round().astype(int)).replace(" ", ""))
        print(a)
        print(r)

def test():
    world = Distribution(Mnist())
    for i in range(10):
        print_traj(world.trajectory(Repeat([0] * 10), i))

test()
assert timer.t() < 0.2
