
import mandalka

from .base import Value, FloatValue

@Value.builder
@mandalka.node
class Gauss(FloatValue):
    def __init__(self, shape):
        import numpy as np

        rng = np.random.RandomState()

        self.get = lambda: rng.randn(*shape)
