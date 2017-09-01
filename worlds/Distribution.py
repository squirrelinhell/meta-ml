
import mandalka

from . import World
from agents import Softmax

@mandalka.node
class Distribution(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.rew_shape

        sm_agents = {}

        def trajectory_batch(agent, seed_batch):
            if agent not in sm_agents:
                sm_agents[agent] = Softmax(agent)

            return world.trajectory_batch(sm_agents[agent], seed_batch)

        self.trajectory_batch = trajectory_batch
