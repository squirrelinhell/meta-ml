
import mandalka

from . import Agent

@mandalka.node
class Constant(Agent):
    def __init__(self, world, seed, value):
        import numpy as np

        value = np.zeros(world.act_shape) + value
        assert value.shape == world.act_shape

        def step(sta_batch, obs_batch):
            tile_dim = [len(obs_batch)] + [1] * len(value.shape)
            return sta_batch, np.tile(value, tile_dim)

        self.step = step
