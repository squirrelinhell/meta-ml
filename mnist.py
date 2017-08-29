#!/usr/bin/env python3

import numpy as np
import tqdm

from worlds import (
    Mnist,
    Distribution,
    Accuracy,
    BasicNet
)

import debug

def train():
    world = BasicNet(
        Distribution(Mnist())
    )

    ep = world.start_episode(123)

    reward = ep.step(np.zeros(world.a_shape))
    for obs, _ in zip(ep, tqdm.tqdm(range(2000))):
        reward = ep.step(reward * 0.0001)

    return ep

print("Ascending gradient...")
trained = train()

print("Checking accuracy...")
world = Accuracy(Mnist(test=True))
ep = world.start_episode(123)
acc_sum = 0.0
for obs, _ in zip(ep, tqdm.tqdm(range(5000))):
    acc_sum += ep.step(trained.solve(obs))
print("%.1f%%" % (acc_sum/50.0))
