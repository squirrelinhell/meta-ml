#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import Mnist, Distribution
from agents import Repeat

def print_exp(exp):
    for o, a, r in exp:
        print(str(o.round().astype(int)).replace(" ", ""))
        print(a)
        print(r)

world = Distribution(Mnist())
for i in range(10):
    world, exp = world.after_episode(Repeat([0] * 10), i)
    print_exp(exp)
