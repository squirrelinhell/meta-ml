#!/usr/bin/env python3

import mandalka
import numpy as np

from models import Random, BasicNet
from problems import Gym, Mnist, CrossEntropy, Accuracy
import debug

def run_episode(world, model):
    episode = world.start_episode()
    steps = 0

    reward_sum = 0.0
    while True:
        inp = episode.next_input()
        if inp is None:
            break

        steps += 1
        out = model.predict(inp)
        reward, grad = episode.next_reward(out)
        reward_sum += reward

    return reward_sum, steps

print("Accuracy of random answers...")
acc = run_episode(Accuracy(Mnist()), Random(Mnist(), 123))
print("%.5f" % (acc[0] / acc[1]))

print("Training network...")
model = BasicNet(
    CrossEntropy(Mnist()),
    seed=123,
    episodes=2
)
mandalka.evaluate(model)

print("Accuracy on training data...")
acc = run_episode(Accuracy(Mnist()), model)
print("%.5f" % (acc[0] / acc[1]))
