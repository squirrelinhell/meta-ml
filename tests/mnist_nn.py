#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import sys
import mandalka

from worlds import Mnist, Accuracy, Distribution, PolicyNet

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def score(world, agent, n_episodes=100):
    total_sum = 0.0
    for i in range(n_episodes):
        ep_sum = 0.0
        ep_samples = 0
        _, exp = test.after_episode(agent, i)
        for o, a, r in exp:
            ep_sum += np.mean(r)
            ep_samples += 1
        total_sum += ep_sum / ep_samples
    return total_sum / n_episodes

test = Accuracy(Mnist(test=True))
train = Distribution(Mnist())

@mandalka.node
class RandomAnswer:
    def action_batch(self, o_batch):
        ans = np.zeros((len(o_batch), 10))
        idx = np.stack((
            np.arange(len(ans)),
            np.random.choice(10, size=len(ans))
        )).T
        ans[idx] = 1.0
        return ans

acc = score(test, RandomAnswer())
sys.stderr.write("Random choice accuracy: %.1f%%\n" % (acc * 100.0))
assert acc > 0.05
assert acc < 0.15

@mandalka.node
class GradAscend:
    def action_batch(self, o_batch):
        upd = o_batch * 200.0
        sys.stderr.write(
            "Gradient norm: %.8f, update: %.8f\n"
                % (norm(o_batch), norm(upd))
        )
        return upd

policy = PolicyNet(train, ep_len=500)
policy, _ = policy.after_episode(GradAscend(), 0)

acc = score(test, policy)
sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc * 100.0))
assert acc > 0.9
assert acc < 1.0
