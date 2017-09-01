
import timer
import sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import (
    Gym, Reinforce, Distribution, WholeTrajectories, PolicyNet
)
from agents import Agent, RandomChoice

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

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

class GradAscend(Agent):
    def __init__(self, world, lr):
        assert world.rew_shape == world.act_shape
        values = np.random.randn(*world.act_shape)

        def learn(traj):
            nonlocal values
            for o, a, r in traj:
                upd = r * lr
                sys.stderr.write("Update norm: %.8f\n" % norm(upd))
                values += upd

        self.action = lambda _: values
        self.learn = learn

def train_policy(problem):
    policy_world = PolicyNet(
        Distribution(Reinforce(WholeTrajectories(problem))),
        batch_size=16
    )
    params = GradAscend(policy_world, 2000)
    for i in range(8):
        params.learn(policy_world.trajectory(params, i))

    return policy_world.build_agent(params)

def test1():
    world = Gym("CartPole-v1")

    s = score(RandomChoice(world))
    sys.stderr.write("Reward/episode (random agent): %.5f\n" % s)

    print("Random agent sanity check:", s >= 15.0, s <= 30.0)

def test2():
    world = Gym("CartPole-v1")
    scores = []

    policy = train_policy(world)
    s = score(policy)
    sys.stderr.write("Reward/episode (policy network): %.5f\n" % s)

    print("Policy network sanity check:", s >= 50.0)

test1()
test2()
assert timer.t() < 8.0
