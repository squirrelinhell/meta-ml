#!/usr/bin/env python3

import mandalka
import numpy as np

from models import Random, BasicNet
from problems import Gym, Mnist, CrossEntropy, Accuracy
import debug

def run_episode(world, model, max_steps=-1):
    episode = world.start_episode()
    steps = 0

    reward_sum = 0.0
    while steps != max_steps:
        inp = episode.next_input()
        if inp is None:
            break

        steps += 1
        out = model.predict(inp)
        reward, grad = episode.next_reward(out)
        reward_sum += reward

    return reward_sum, steps

model = BasicNet(
    CrossEntropy(Mnist()),
    seed=123,
    steps=500000
)

acc = run_episode(Accuracy(Mnist()), model, max_steps=10000)
print("Accuracy on training data: %.5f" % (acc[0] / acc[1]))
