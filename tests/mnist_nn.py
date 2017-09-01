
import timer
import sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import Mnist, Accuracy, Distribution, PolicyNet
from agents import Agent, RandomChoice
assert timer.t() < 0.15

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def score(agent, n_episodes=1000):
    world = Accuracy(Mnist(test=True))
    rew_sum = 0.0
    for traj in world.trajectory_batch(agent, range(n_episodes)):
        for o, a, r in traj:
            rew_sum += np.mean(r)
    return rew_sum / n_episodes

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

def test():
    train = Distribution(Mnist())

    acc = score(RandomChoice(train))
    sys.stderr.write("Random choice accuracy: %.1f%%\n" % (acc * 100.0))
    print("Random choice sanity check:", acc > 0.05, acc < 0.15)

    policy_world = PolicyNet(train, batch_size=128)
    params = GradAscend(policy_world, 250)
    for i in range(200):
        params.learn(policy_world.trajectory(params, i))

    acc = score(policy_world.get_policy(params))

    sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc * 100.0))
    print("Neural net sanity check:", acc > 0.9, acc < 1.0)

test()
assert timer.t() < 6.0
