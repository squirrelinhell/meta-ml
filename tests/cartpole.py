
import timer
import os
import sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import *
from agents import *
assert timer.t() < 0.15

def score(agent, n_episodes=128, batch_size=16):
    world = Gym("CartPole-v1")
    rew_sum = 0.0
    done = 0
    while done < n_episodes:
        for traj in world.trajectory_batch(agent, range(batch_size)):
            for o, a, r in traj:
                rew_sum += np.mean(r)
        done += batch_size
    return rew_sum / done

class LearningRate(Agent):
    def __init__(self, magnitude=0.8):
        def action(o):
            sys.stderr.write("Gradient fitness: %6.2f\n" % o[0])
            return [magnitude]
        self.action = action

def solve(problem):
    problem = WholeTrajectories(problem)
    problem = Distribution(Reinforce(problem))
    problem = PolicyNet(problem, batch_size=16)
    problem = GradAscent(problem, n_steps=8)
    return problem.inner_agent(LearningRate(), 0)

def test1():
    world = Gym("CartPole-v1")

    s = score(RandomChoice(world))
    sys.stderr.write("Reward/episode (random agent): %.5f\n" % s)

    print("Random agent sanity check:", s >= 15.0, s <= 30.0)

def test2():
    world = Gym("CartPole-v1")
    scores = []

    policy = solve(world)
    s = score(policy)
    sys.stderr.write("Reward/episode (policy network): %.5f\n" % s)

    print("Policy network sanity check:", s >= 50.0)

    if "DEBUG" in os.environ:
        for _ in range(5):
            world.render(policy)
    else:
        assert timer.t() < 8.0

test1()
test2()
