
import timer
import sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import *
from agents import *
assert timer.t() < 0.15

def score(agent, n_episodes=1000):
    world = Accuracy(Mnist(test=True))
    rew_sum = 0.0
    for traj in world.trajectory_batch(agent, range(n_episodes)):
        for o, a, r in traj:
            rew_sum += np.mean(r)
    return rew_sum / n_episodes

class LearningRate(Agent):
    def __init__(self, magnitude=1.1):
        def action(o):
            sys.stderr.write("Gradient fitness: %6.2f\n" % o[0])
            return [magnitude]
        self.action = action

def solve(problem):
    problem = Distribution(problem)
    problem = PolicyNet(problem, batch_size=128)
    problem = GradAscent(problem, n_steps=200)
    return problem.inner_agent(LearningRate(), 0)

def test():
    acc = score(RandomChoice(Mnist()))
    sys.stderr.write("Random choice accuracy: %.1f%%\n" % (acc * 100.0))
    print("Random choice sanity check:", acc > 0.05, acc < 0.15)

    policy = solve(Mnist())
    acc = score(policy)
    sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc * 100.0))
    print("Neural net sanity check:", acc > 0.9, acc < 1.0)

test()
assert timer.t() < 6.0
