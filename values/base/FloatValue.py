
from . import Value
from agents import Agent

class FloatValue(Value, Agent):
    def step(self, states, observations):
        value = self.get()
        if len(observations) == 1:
            return states, value.reshape((1,) + value.shape)
        else:
            import numpy as np
            tile_dim = (len(observations),) + (1,) * len(value.shape)
            return states, np.tile(value, tile_dim)
