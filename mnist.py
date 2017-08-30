#!/usr/bin/env python3

import numpy as np
import tqdm

from worlds import (
    Mnist,
    Distribution,
    Accuracy,
    PolicyNet
)

import debug

def train():
    world = PolicyNet(
        Distribution(Mnist())
    )

    ep = world.start_episode(123)

    reward = ep.step(np.zeros(world.a_shape))
    for _ in range(500):
        print("%.8f" % np.mean(np.abs(reward)))
        reward = ep.step(reward * 200.0)

    return ep

print("Ascending gradient...")
trained = train()

print("Checking accuracy...")
world = Accuracy(Mnist(test=True))
acc_sum = 0.0
for i in tqdm.tqdm(range(5000)):
    ep = world.start_episode(i)
    acc_sum += ep.step(trained.solve(ep.get_observation()))
print("%.1f%%" % (acc_sum/50.0))
