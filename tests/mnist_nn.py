#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import sys
import mandalka

from worlds import Mnist, Accuracy, Batch, Distribution, PolicyNet

train = Distribution(Batch(Mnist(), batch_size=128))

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def score(agent):
    test = Accuracy(Mnist(test=True))
    acc_sum = 0.0
    n_samples = 0
    for i in range(1000):
        _, exp = test.after_episode(agent, i)
        for o, a, r in exp:
            acc_sum += np.mean(r)
            n_samples += 1
    return acc_sum / n_samples

@mandalka.node
class RandomAnswer:
    def action_batch(self, o_batch):
        ans = np.zeros(10)
        ans[np.random.choice(10)] = 1.0
        return np.asarray([ans])

acc = score(RandomAnswer())
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

acc = score(policy)
sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc * 100.0))
assert acc > 0.9
assert acc < 1.0
