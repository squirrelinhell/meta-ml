
from . import Agent

class Repeat(Agent):
    def __init__(self, *values):
        import numpy as np

        values = np.array(values, dtype=np.float32)
        i = -1

        def action(_):
            nonlocal i
            i = (i + 1) % len(values)
            return values[i]

        self.action = action
