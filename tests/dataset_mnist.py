
from tests._ import timer

import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import Mnist
from agents import Zero, Softmax

def print_traj(t):
    for o, a, r in t:
        print(str(o.round().astype(int)).replace(" ", ""))
        print(a)
        print(r)

def test():
    agent = Softmax(Mnist(), 0, logits=Zero)
    for i in range(3):
        print_traj(Mnist().trajectory(agent, i))
    for i in range(3):
        print_traj(Mnist(test=True).trajectory(agent, i))

test()
assert timer.t() < 0.2
