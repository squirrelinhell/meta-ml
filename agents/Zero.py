
import mandalka

from . import Agent

@mandalka.node
class Zero:
    def __init__(self, world, seed):
        import numpy as np

        def step(sta_batch, obs_batch):
            act_batch = np.zeros((len(obs_batch),) + world.act_shape)
            return sta_batch, act_batch

        self.step = step
