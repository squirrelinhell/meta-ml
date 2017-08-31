
import mandalka

from . import World
from agents import Softmax

@mandalka.node
class Distribution(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.o_shape
        self.get_action_shape = lambda: world.a_shape
        self.get_reward_shape = lambda: world.r_shape

        sm_agents = {}

        def after_episode(agent, seed):
            if agent not in sm_agents:
                sm_agents[agent] = Softmax(agent)

            w2, exp = world.after_episode(sm_agents[agent], seed)

            return self if w2 == world else Distribution(w2), exp

        self.after_episode = after_episode
