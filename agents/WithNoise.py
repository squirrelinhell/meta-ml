
import mandalka

from .base import Agent, WrapperAgent
from worlds import World
from values import Value

@Agent.builder
@mandalka.node
class WithNoise(WrapperAgent):
    def __init__(self, world, seed, value, stddev):
        import numpy as np
        seed = Agent.split_seed(seed)

        stddev = Value.get_float(stddev, world.act_shape, seed())

        world = WithNoiseWorld(world, stddev)
        value = Agent.build(value, world, seed())

        # Get some really unpredictable noise (ignore seed)
        rng = np.random.RandomState()

        def process_action(a):
            return a + np.multiply(rng.randn(*a.shape), stddev.get())

        super().__init__(value, process_action=process_action)

@mandalka.node
class WithNoiseWorld(World):
    def __init__(self, world, stddev):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.rew_shape

        def trajectory_batch(agent, seed_batch):
            assert isinstance(agent, Agent)

            return world.trajectory_batch(
                WithNoise(world, 0, value=agent, stddev=stddev),
                seed_batch
            )

        self.trajectory_batch = trajectory_batch
