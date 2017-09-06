
import os
import sys
import numpy as np
import tensorflow as tf
tf.Session().run(tf.placeholder_with_default(0, shape=None))

from test_setup import timer
from worlds import *
from agents import *
from values import *
assert timer() < 0.05

world = Gym("CartPole-v1")

def score(agent):
    rew_sum = 0.0
    for _ in range(2):
        for t in world.trajectory_batch(agent, range(16)):
            for o, a, r in t:
                rew_sum += np.mean(r)
    return rew_sum / 32.0

def test1():
    agent = RandomChoice(world, 123, p=0.5)
    s = score(agent)
    sys.stderr.write("Reward/episode (random agent): %.5f\n" % s)
    print("Random agent sanity check:", s >= 15.0, s <= 30.0)

def test2():
    SupervisedAgent = Softmax(
        logits=BasicNet(
            hidden_layers=[32],
            batch_size=16,
            params=GradAscent(n_steps=8, log_lr=0.9, init=Gauss)
        )
    )
    RLAgent = WholeTrajectories(
        agent=RandomChoice(
            p=Reinforce(agent=SupervisedAgent)
        )
    )
    agent = RLAgent(world, 123)
    s = score(agent)
    sys.stderr.write("Reward/episode (policy network): %.5f\n" % s)
    print("Policy network sanity check:", s >= 50.0)

    if "DEBUG" in os.environ:
        timer()
        for _ in range(5):
            world.render(agent)
    else:
        assert timer() < 4.0

test1()
test2()
