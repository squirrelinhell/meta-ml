
import mandalka

from . import World
from agents import AsDistribution

@mandalka.node
class Distribution(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.o_shape
        self.get_action_shape = lambda: world.a_shape
        self.get_reward_shape = lambda: world.r_shape

        dist_agents = {}

        def after_episode(agent, seed):
            if agent not in dist_agents:
                dist_agents[agent] = AsDistribution(agent)

            w2, exp = world.after_episode(dist_agents[agent], seed)

            return self if w2 == world else Distribution(w2), exp

        self.after_episode = after_episode
