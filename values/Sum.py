
import mandalka

from .base import Value, FloatValue

@Value.builder
@mandalka.node
class Sum(FloatValue):
    def __init__(self, shape, value1, value2):
        value1 = Value.get_float(value1, shape)
        value2 = Value.get_float(value2, shape)

        self.get = lambda: value1.get() + value2.get()
