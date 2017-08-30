#!/usr/bin/env python3

import sys
import numpy as np

from worlds import Mnist, Accuracy, Distribution, PolicyNet

policy = PolicyNet(Distribution(Mnist()))

policy_ep = policy.start_episode(123)
grad = policy_ep.step(np.zeros(policy.a_shape))
for _ in range(500):
    grad = policy_ep.step(grad * 200.0)

test = Accuracy(Mnist(test=True))

acc_sum = 0.0
for i in range(1000):
    test_ep = test.start_episode(i)
    for obs in test_ep:
        acc_sum += test_ep.step(np.random.randn(10)) / 1000.0

sys.stderr.write("Random choice accuracy: %.1f%%\n" % (acc_sum * 100.0))
assert acc_sum < 0.15
assert acc_sum > 0.05

acc_sum = 0.0
for i in range(5000):
    test_ep = test.start_episode(i)
    for obs in test_ep:
        acc_sum += test_ep.step(policy_ep.solve(obs)) / 5000.0

sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc_sum * 100.0))
assert acc_sum < 1.0
assert acc_sum > 0.9
