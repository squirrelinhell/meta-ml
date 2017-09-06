
from . import Value
from agents import Agent

class FloatValue(Value, Agent):
    # def __init__(self, shape, ...)

    # def get(self):
    #     ...

    # Implement a stateless Agent based on Value.get()

    def step(self, states, observations):
        if len(observations) == 1:
            value = self.get()
            return states, value.reshape((1,) + value.shape)
        else:
            import numpy as np
            return states, np.array([self.get() for _ in observations])
