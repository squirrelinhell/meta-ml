
import timer
import sys
import mandalka
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from worlds import Mnist, Accuracy, Distribution, Batch, PolicyNet
from agents import Agent, RandomChoice
assert timer.t() < 0.1

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def score(agent, n_episodes=1000):
    world = Accuracy(
        Batch(Mnist(test=True), n_episodes)
    )
    rew_sum = 0.0
    for o, a, r in world.after_episode(agent, 0)[1]:
        rew_sum += np.mean(r)
    return rew_sum / n_episodes

@mandalka.node
class GradAscend(Agent):
    lr = 250.0

    def action_batch(self, o_batch):
        upd = o_batch * GradAscend.lr
        sys.stderr.write(
            "Gradient norm: %.8f, update: %.8f\n"
                % (norm(o_batch), norm(upd))
        )
        return upd

def test():
    train = Distribution(Batch(Mnist(), 128))

    acc = score(RandomChoice(train))
    sys.stderr.write("Random choice accuracy: %.1f%%\n" % (acc * 100.0))
    print("Random choice sanity check:", acc > 0.05, acc < 0.15)

    policy = PolicyNet(train, init_seed=1, ep_len=200)
    policy, _ = policy.after_episode(GradAscend(), 0)

    acc = score(policy)
    sys.stderr.write("Neural net accuracy: %.1f%%\n" % (acc * 100.0))
    print("Neural net sanity check:", acc > 0.9, acc < 1.0)

test()
assert timer.t() < 6.0
