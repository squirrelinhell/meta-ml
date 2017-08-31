
import mandalka

from . import Agent

@mandalka.node
class Repeat(Agent):
    def __init__(self, *values):
        import numpy as np

        values = np.array(values, dtype=np.float32)
        i = -1

        def next_val():
            nonlocal i
            i = (i + 1) % len(values)
            return values[i]

        self.action_batch = lambda o: np.array([next_val() for _ in o])
