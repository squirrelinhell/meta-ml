#!/usr/bin/env python3

import numpy as np

from models import Random
from problems import Gym, Mnist
import debug

def run_episode(world, model):
    episode = world.start_episode()

    reward_sum = 0.0
    while True:
        inp = episode.next_input()
        if inp is None:
            break

        out = model.predict(inp)
        reward, grad = episode.next_reward(out)
        reward_sum += reward

    return reward_sum

world = Gym("CartPole-v1")
model = Random(world, seed=123)

for _ in range(10):
    print(run_episode(world, model))

world = Mnist()
model = Random(world, seed=123)
ep = world.start_episode()

inp = ep.next_input()
print((inp*2).astype(np.int))

out = model.predict(inp)
print(np.round(out, 2))

reward, grad = ep.next_reward(out)
print(reward)
print(np.round(grad, 2))
