
import timer
import sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import *
from agents import *
assert timer.t() < 0.15

def solve(problem):
    problem = Batch(problem, batch_size=128)
    problem = BasicNet(Distribution(problem), hidden_layers=[128])
    problem = GradAscent(problem, n_steps=200)
    return problem.inner_agent(Repeat([1.1]), 0)

def score(agent, n_episodes=1000):
    world = Accuracy(Batch(Mnist(test=True), n_episodes))
    rew_sum = 0.0
    for o, a, r in world.trajectory(agent, 0):
        rew_sum += np.mean(r)
    return rew_sum / n_episodes

def test1():
    agent = RandomChoice(Mnist())
    acc = score(agent)
    sys.stderr.write("Random choice accuracy: %.1f%%\n" % (acc * 100.0))
    print("Random choice sanity check:", acc > 0.05, acc < 0.15)

def test2():
    agent = solve(Mnist())
    acc = score(agent)
    sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc * 100.0))
    print("Neural net sanity check:", acc > 0.9, acc < 1.0)

test1()
test2()
assert timer.t() < 6.0
