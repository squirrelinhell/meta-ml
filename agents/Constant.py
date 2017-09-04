
import mandalka

from .base import Agent, StatelessAgent

@Agent.builder
@mandalka.node
class Constant(StatelessAgent):
    def __init__(self, world, seed, value):
        import numpy as np

        assert seed == 0
        del seed

        value = np.zeros(world.act_shape) + value
        assert value.shape == world.act_shape
        value.setflags(write=False)
        super().__init__(get_action=lambda _: value)
