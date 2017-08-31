
import mandalka

from . import Agent

@mandalka.node
class Softmax(Agent):
    def __init__(self, agent):
        import numpy as np

        def action_batch(o_batch):
            a_batch = agent.action_batch(o_batch)

            a_batch = np.array(a_batch)
            for i, a in enumerate(a_batch):
                # Safely calculate softmax
                a = np.exp(a - np.amax(a))
                a_batch[i] = a / a.sum()

            return a_batch

        self.action_batch = action_batch
