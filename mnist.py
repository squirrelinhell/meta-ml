#!/usr/bin/env python3

from datasets import Mnist
from problems import CrossEntropy, Accuracy
from models import BasicNet
import debug

def run_episode(world, model):
    episode = world.start_episode()
    steps = 0

    reward_sum = 0.0
    for inp in episode:
        steps += 1
        out = model.predict(inp)
        reward, _ = episode.next_reward(out)
        reward_sum += reward

    return reward_sum, steps

model = BasicNet(
    CrossEntropy(Mnist()),
    seed=123
)

acc = run_episode(
    Accuracy(Mnist(test=True)),
    model
)

print("Accuracy on test data: %.5f" % (acc[0] / acc[1]))
