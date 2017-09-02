
import mandalka

from . import Agent

@mandalka.node
class Repeat(Agent):
    def __init__(self, world, seed, values):
        import numpy as np

        values = np.array(values, dtype=np.float32)
        assert values.shape[1:] == world.act_shape

        i = -1
        def next_action():
            nonlocal i
            i = (i + 1) % len(values)
            return values[i]

        def step(sta_batch, obs_batch):
            act_batch = np.array([next_action() for _ in obs_batch])
            return sta_batch, act_batch

        self.step = step
