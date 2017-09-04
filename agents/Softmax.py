
import mandalka

from .base import Agent, WrapperAgent
from worlds import World

@Agent.builder
@mandalka.node
class Softmax(WrapperAgent):
    def __init__(self, world, seed, logits):
        import numpy as np

        logits = Agent.build(logits, SoftmaxWorld(world), seed)
        del seed

        def process_action(a):
            a = np.exp(a - np.amax(a))
            return a / a.sum()

        super().__init__(logits, process_action=process_action)

@mandalka.node
class SoftmaxWorld(World):
    def __init__(self, world):
        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.rew_shape

        def trajectory_batch(agent, seed_batch):
            assert isinstance(agent, Agent)

            return world.trajectory_batch(
                Softmax(world, 0, logits=agent),
                seed_batch
            )

        self.trajectory_batch = trajectory_batch
