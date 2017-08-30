#!/usr/bin/env python3

import numpy as np

from worlds import Mnist, Distribution, Reinforce, Accuracy

for i in range(10):
    ep = Mnist().start_episode(i)
    for obs in ep:
        print((obs * 1.9).astype(int))
        print()
        print(ep.step(np.zeros(10)).astype(int))
        print()

for i in range(20):
    ep = Distribution(Mnist()).start_episode(i)
    for obs in ep:
        print((ep.step(np.zeros(10)) * 100.0).astype(int))

print()

for i in range(20):
    ep = Reinforce(Accuracy(Mnist())).start_episode(i)
    for obs in ep:
        print((ep.step(np.zeros(10)) * 100.0).astype(int))
