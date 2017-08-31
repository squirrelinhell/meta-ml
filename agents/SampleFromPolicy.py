
import mandalka

from . import Agent

@mandalka.node
class SampleFromPolicy(Agent):
    def __init__(self, agent):
        import numpy as np

        rng = np.random.RandomState()

        def action_batch(o_batch):
            a_batch = np.asarray(agent.action_batch(o_batch))
            assert len(a_batch.shape) == 2

            ans = np.zeros(a_batch.shape)
            for i, ps in enumerate(a_batch):
                ans[i, np.random.choice(len(ps), p=ps)] = 1.0

            return ans

        self.action_batch = action_batch
