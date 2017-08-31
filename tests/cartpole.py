#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import sys
import mandalka

from worlds import Gym, Reinforce, Distribution, PolicyNet
from agents import Agent, RandomChoice, SampleFromPolicy, Softmax

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def score(agent, n_episodes=3, batch_size=16):
    world = Gym(
        "CartPole-v1",
        batch_size=batch_size,
        normalize_rewards=False
    )
    rew_sum = 0.0
    for i in range(n_episodes):
        _, exp = world.after_episode(agent, i)
        for o, a, r in exp:
            rew_sum += np.mean(r)
    return rew_sum / (n_episodes * batch_size)

@mandalka.node
class GradAscend(Agent):
    lr = 2000.0

    def action_batch(self, o_batch):
        upd = o_batch * GradAscend.lr
        sys.stderr.write(
            "Gradient norm: %.8f, update: %.8f\n"
                % (norm(o_batch), norm(upd))
        )
        return upd

def train_policy(problem, seed):
    policy = PolicyNet(
        Distribution(Reinforce(problem)),
        init_seed=1,
        ep_len=6
    )
    policy, _ = policy.after_episode(GradAscend(), seed)
    return SampleFromPolicy(Softmax(policy))

def test1():
    world = Gym("CartPole-v1")

    s = score(RandomChoice(world))
    sys.stderr.write("Reward/episode (random agent): %.5f\n" % s)

    print("Random agent sanity check:", s >= 16.0, s <= 26.0)

def test2():
    world = Gym("CartPole-v1")
    scores = []

    for i in range(3):
        s = score(train_policy(world, i))
        scores.append(s)
        sys.stderr.write("Reward/episode (policy network): %.5f\n" % s)

    scores = sorted(scores)
    print("Policy network sanity check:", scores[1] >= 60.0)

test1()
test2()
