
import mandalka

from . import Agent

@mandalka.node
class Cycle(Agent):
    def __init__(self, world, seed, sequence):
        assert seed == 0
        del seed

        import numpy as np

        sequence = np.array(sequence, dtype=np.float32)
        assert sequence.shape[1:] == world.act_shape

        def step(sta_batch, obs_batch):
            sta_batch = [
                [0] if i is None else [(i[0] + 1) % len(sequence)]
                for i in sta_batch
            ]
            act_batch = np.array([sequence[i[0]] for i in sta_batch])
            return sta_batch, act_batch

        self.step = step
