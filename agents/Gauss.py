
import mandalka

from .base import Agent, StatelessAgent

@Agent.builder
@mandalka.node
class Gauss(StatelessAgent):
    def __init__(self, world, seed):
        import numpy as np

        rng = np.random.RandomState(seed)
        del seed

        value = rng.randn(*world.act_shape)
        value.setflags(write=False)
        super().__init__(get_action=lambda _: value)
