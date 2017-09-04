
import mandalka

from . import Agent

@Agent.builder
@mandalka.node
class Gauss(Agent):
    def __init__(self, world, seed):
        import numpy as np
        rng = np.random.RandomState(seed)
        del seed

        values = rng.randn(*world.act_shape)

        def step(sta_batch, obs_batch):
            tile_dim = [len(obs_batch)] + [1] * len(values.shape)
            return sta_batch, np.tile(values, tile_dim)

        self.step = step
