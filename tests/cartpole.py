
import timer
import os
import sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import *
from agents import *
assert timer.t() < 0.15

def solve(problem):
    problem = Reinforce(Policy(problem, batch_size=16))
    problem = BasicNet(Distribution(problem), hidden_layers=[128])
    problem = GradAscent(problem, n_steps=8)
    return problem.inner_agent(Repeat([0.8]), 0)

def score(agent):
    world = Batch(Batch(Gym("CartPole-v1"), 16), 8) # 128 episodes
    rew_sum = 0.0
    for o, a, r in world.trajectory(agent, 0):
        rew_sum += np.mean(r)
    return rew_sum / 128.0

def test1():
    agent = RandomChoice(Gym("CartPole-v1"))
    s = score(agent)
    sys.stderr.write("Reward/episode (random agent): %.5f\n" % s)
    print("Random agent sanity check:", s >= 15.0, s <= 30.0)

def test2():
    agent = solve(Gym("CartPole-v1"))
    s = score(agent)
    sys.stderr.write("Reward/episode (policy network): %.5f\n" % s)
    print("Policy network sanity check:", s >= 50.0)

    if "DEBUG" in os.environ:
        timer.t()
        for _ in range(5):
            Gym("CartPole-v1").render(agent)
    else:
        assert timer.t() < 9.0

test1()
test2()
