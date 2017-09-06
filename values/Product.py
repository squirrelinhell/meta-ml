
import mandalka

from .base import Value, FloatValue

@Value.builder
@mandalka.node
class Product(FloatValue):
    def __init__(self, shape, value1, value2):
        import numpy as np

        value1 = Value.get_float(value1, shape)
        value2 = Value.get_float(value2, shape)

        self.get = lambda: np.multiply(value1.get(), value2.get())
