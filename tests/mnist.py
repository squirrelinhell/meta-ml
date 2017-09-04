
from tests._ import timer

import sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import *
from agents import *
assert timer.t() < 0.15

def score(agent, n_episodes=1000):
    world = Accuracy(Mnist(test=True))
    rew_sum = 0.0
    for t in world.trajectory_batch(agent, range(n_episodes)):
        for o, a, r in t:
            rew_sum += np.mean(r)
    return rew_sum / n_episodes

def test1():
    agent = RandomChoice(Mnist(), 123, p=0.1)
    acc = score(agent)
    sys.stderr.write("Random choice accuracy: %.1f%%\n" % (acc * 100.0))
    print("Random choice sanity check:", acc > 0.05, acc < 0.15)

def test2():
    SupervisedAgent = Softmax(
        logits=BasicNet(
            hidden_layers=[128],
            batch_size=128,
            params=GradAscent(n_steps=200, log_lr=1.1)
        )
    )
    agent = SupervisedAgent(Mnist(), 0)
    acc = score(agent)
    sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc * 100.0))
    print("Neural net sanity check:", acc > 0.9, acc < 1.0)

test1()
test2()
assert timer.t() < 6.0
