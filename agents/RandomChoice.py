
import mandalka

from .base import Agent, WrapperAgent
from worlds import World

@Agent.builder
@mandalka.node
class RandomChoice(WrapperAgent):
    def __init__(self, world, seed, p):
        import numpy as np

        p = Agent.build(p, RandomChoiceWorld(world), seed)
        del seed

        # Get some really unpredictable choices (ignore seed)
        rng = np.random.RandomState()

        def process_action(a):
            assert len(a.shape) == 1
            i = rng.choice(len(a), p=a)
            a[:] = 0.0
            a[i] = 1.0
            return a

        super().__init__(p, process_action=process_action)

@mandalka.node
class RandomChoiceWorld(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.rew_shape

        def trajectory_batch(agent, seed_batch):
            rng = np.random.RandomState(seed_batch[0])
            seed_batch[0] = rng.randint(2**32)

            return world.trajectory_batch(
                RandomChoice(world, rng.randint(2**32), p=agent),
                seed_batch
            )

        self.trajectory_batch = trajectory_batch
