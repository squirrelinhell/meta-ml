#!/usr/bin/env python3

import numpy as np

from models import Random
from problems import Gym, Mnist
import debug

def evaluate(model, episode):
    reward_sum = 0.0
    while True:
        inps = episode.get_input_batch()
        if inps is None:
            break

        outs = model.predict_batch(inps)
        assert len(inps) == len(outs)

        rewards = episode.get_reward_batch(outs)
        assert len(inps) == rewards.size

        reward_sum += np.mean(rewards)

    return reward_sum

world = Gym("CartPole-v1")
model = Random(world, seed=123)

for _ in range(10):
    print(evaluate(model, world.start_episode()))

world = Mnist()
model = Random(world, seed=123)
ep = world.start_episode()
batch = ep.get_input_batch(2)
print(batch)

pred = model.predict_batch(batch)
print(pred)

grad = ep.get_reward_gradient(pred)
print(grad)
