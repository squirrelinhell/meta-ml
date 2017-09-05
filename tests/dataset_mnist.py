
import numpy as np

from test_setup import timer
from worlds import Mnist
from agents import Softmax

def print_traj(t):
    for o, a, r in t:
        print(str(o.round().astype(int)).replace(" ", ""))
        print(a)
        print(r)

def test():
    agent = Softmax(Mnist(), 0, logits=0.0)
    for i in range(3):
        print_traj(Mnist().trajectory(agent, i))
    for i in range(3):
        print_traj(Mnist(test=True).trajectory(agent, i))

test()
assert timer() < 0.2
