
import mandalka

from .base import Value, FloatValue
from agents import Agent

@Value.builder
@mandalka.node
class Constant(FloatValue):
    def __init__(self, shape, value):
        import numpy as np

        if isinstance(value, Agent):
            _, (value,) = value.step([None], [None])
        else:
            value = np.zeros(shape, dtype=np.float32) + value

        assert value.shape == shape
        value.setflags(write=False)
        self.get = lambda: value
