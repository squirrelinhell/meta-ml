
import mandalka

from .base import Value, FloatValue

@Value.builder
@mandalka.node
class Constant(FloatValue):
    def __init__(self, shape, seed, value):
        import numpy as np

        value = np.zeros(shape, dtype=np.float32) + value
        assert value.shape == shape

        value.setflags(write=False)
        self.get = lambda: value
