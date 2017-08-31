
import mandalka

from . import Agent

@mandalka.node
class RandomChoice(Agent):
    def __init__(self, world):
        import numpy as np

        rng = np.random.RandomState()
        a_size = np.prod(world.a_shape)

        def action_batch(o_batch):
            ans = np.zeros((len(o_batch), a_size))

            choices = np.random.choice(a_size, size=len(ans))
            for i, c in enumerate(choices):
                ans[i, c] = 1.0

            return ans.reshape((len(ans),) + world.a_shape)

        self.action_batch = action_batch
