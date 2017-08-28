
import mandalka
import numpy as np

from . import Model

@mandalka.node
class Random(Model):
    def __init__(self, problem, seed):
        rng = np.random.RandomState(seed=seed)

        self.predict = lambda inp: rng.randn(*problem.output_shape)
