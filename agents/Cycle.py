
import mandalka

from .base import Agent

@Agent.builder
@mandalka.node
class Cycle(Agent):
    def __init__(self, world, seed, sequence):
        import numpy as np

        sequence = np.array(sequence, dtype=np.float32)
        assert sequence.shape[1:] == world.act_shape

        def step(states, observations):
            states = [
                [0] if i is None else [(i[0] + 1) % len(sequence)]
                for i in states
            ]
            actions = np.array([sequence[i[0]] for i in states])
            return states, actions

        self.step = step
