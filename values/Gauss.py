
import mandalka

from .base import Value, FloatValue

@Value.builder
@mandalka.node
class Gauss(FloatValue):
    def __init__(self, shape, seed):
        import numpy as np

        value = np.random.RandomState(seed).randn(*shape)

        value.setflags(write=False)
        self.get = lambda: value
