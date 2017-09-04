
import mandalka

from . import Agent

@Agent.builder
@mandalka.node
class RandomChoice(Agent):
    def __init__(self, world, seed, p):
        import numpy as np
        rng = np.random.RandomState(seed)
        del seed

        p = Agent.build(p, world, rng.randint(2**32))

        def step(sta_batch, obs_batch):
            sta_batch, act_batch = p.step(sta_batch, obs_batch)
            act_batch = np.asarray(act_batch)
            assert len(act_batch.shape) == 2

            ans = np.zeros(act_batch.shape)
            for i, ps in enumerate(act_batch):
                ans[i, rng.choice(len(ps), p=ps)] = 1.0

            return sta_batch, ans

        self.step = step
