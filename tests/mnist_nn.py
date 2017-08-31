#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import sys
import mandalka

from worlds import Mnist, Accuracy, Distribution, PolicyNet
from agents import Agent, RandomChoice

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def score(world, agent, n_episodes=100):
    total_sum = 0.0
    for i in range(n_episodes):
        ep_sum = 0.0
        ep_samples = 0
        _, exp = world.after_episode(agent, i)
        for o, a, r in exp:
            ep_sum += np.mean(r)
            ep_samples += 1
        total_sum += ep_sum / ep_samples
    return total_sum / n_episodes

@mandalka.node
class GradAscend(Agent):
    lr = 200.0

    def action_batch(self, o_batch):
        upd = o_batch * GradAscend.lr
        sys.stderr.write(
            "Gradient norm: %.8f, update: %.8f\n"
                % (norm(o_batch), norm(upd))
        )
        return upd

def test():
    test = Accuracy(Mnist(test=True))
    train = Distribution(Mnist())

    acc = score(test, RandomChoice(train))
    sys.stderr.write("Random choice accuracy: %.1f%%\n" % (acc * 100.0))
    print("Random choice sanity check:", acc > 0.05, acc < 0.15)

    policy = PolicyNet(train, init_seed=1, ep_len=500)
    policy, _ = policy.after_episode(GradAscend(), 0)

    acc = score(test, policy)
    sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc * 100.0))
    print("Neural net sanity check:", acc > 0.9, acc < 1.0)

test()
