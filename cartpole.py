#!/usr/bin/env python3

from problems import Gym, Reinforce, ParallelEpisodes
from models import Random, BasicNet
import debug

def run_episode(world, model):
    episode = world.start_episode()
    steps = 0

    reward_sum = 0.0
    for inp in episode:
        steps += 1
        out = model.predict(inp)
        reward, _ = episode.next_reward(out)
        reward_sum += reward

    return reward_sum, steps

for i in range(10):
    score = run_episode(
        Gym("CartPole-v1"),
        Random(Gym("CartPole-v1"), seed=123)
    )

    print("Episode reward (random agent): %.5f (%d steps)" % score)

for i in range(10):
    score = run_episode(
        Gym("CartPole-v1"),
        BasicNet(
            Reinforce(
                ParallelEpisodes(Gym("CartPole-v1"))
            ),
            batch_size=32,
            steps=50000,
            hidden_layers=[8],
            lr="0.0001",
            seed=123
        )
    )

    print("Episode reward (policy gradient WIP): %.5f (%d steps)" % score)
