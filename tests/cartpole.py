
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

def score(agent):
    world = Gym("CartPole-v1")
    rew_sum = 0.0
    for _ in range(10):
        for t in world.trajectory_batch(agent, range(10)):
            for o, a, r in t:
                rew_sum += np.mean(r)
    return rew_sum / 100.0

def test1():
    agent = RandomChoice(Gym("CartPole-v1"), 123, p=0.5)
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
        agent=Reinforce(policy=SupervisedAgent)
    )
    agent = RLAgent(Gym("CartPole-v1"), 123)
    s = score(agent)
    sys.stderr.write("Reward/episode (policy network): %.5f\n" % s)
    print("Policy network sanity check:", s >= 50.0)

    if "DEBUG" in os.environ:
        timer()
        for _ in range(5):
            Gym("CartPole-v1").render(agent)
    else:
        assert timer() < 8.0

test1()
test2()
