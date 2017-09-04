
import mandalka

from .base import Agent, WrapperAgent

@Agent.builder
@mandalka.node
class RandomChoice(WrapperAgent):
    def __init__(self, world, seed, p):
        import numpy as np

        p = Agent.build(p, world, seed)
        del seed

        # Get some really unpredictable choices (ignore seed)
        rng = np.random.RandomState()

        def process_action(a):
            assert len(a.shape) == 1
            i = rng.choice(len(a), p=a)
            a[:] = 0.0
            a[i] = 1.0
            return a

        super().__init__(p, process_action=process_action)
