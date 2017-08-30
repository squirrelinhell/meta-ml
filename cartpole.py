#!/usr/bin/env python3

import numpy as np
import tqdm

from worlds import Gym, Reinforce, PolicyNet

import debug

def train_policy(problem):
    world = PolicyNet(
        Reinforce(problem),
        normalize_rewards=True,
        batch_size=20
    )
    ep = world.start_episode(123)

    reward = 0.01 * ep.step(np.zeros(world.a_shape))
    for _, ep_len in zip(range(50), ep):
        ep_len = int(ep_len[0])
        print("=" * ep_len, ep_len)
        reward *= 0.99
        reward += 0.01 * ep.step(reward * -10.0) # ?? should be positive

    return ep

def score(world, policy):
    avg = 0.0
    for i in range(10):
        ep = world.start_episode(i)
        avg += 0.1 * np.sum([
            np.mean(ep.step(policy(inp)))
            for inp in ep
        ])
    return avg

world = Gym("CartPole-v1")

print("Average reward (random agent): %.5f" % score(
    world,
    lambda _: np.random.randn(*world.a_shape)
))

for _ in range(10):
    agent = train_policy(world)

    print("Average reward (policy network): %.5f" % score(
        Reinforce(world, test=True),
        agent.solve
    ))
