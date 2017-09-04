
import mandalka

from . import Agent
from worlds import World

@Agent.builder
@mandalka.node
class Softmax(Agent):
    def __init__(self, world, seed, logits):
        import numpy as np

        logits = Agent.build(logits, SoftmaxWorld(world), seed)

        def step(sta_batch, obs_batch):
            sta_batch, act_batch = logits.step(sta_batch, obs_batch)

            act_batch = np.asarray(act_batch)
            for i, a in enumerate(act_batch):
                # Safely calculate softmax
                a = np.exp(a - np.amax(a))
                act_batch[i] = a / a.sum()

            return sta_batch, act_batch

        self.step = step

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
