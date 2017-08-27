
import mandalka
import numpy as np

from . import Model

@mandalka.node
class Random(Model):
    def __init__(self, problem, seed):
        rng = np.random.RandomState(seed=seed)

        self.predict_batch = lambda input_batch: rng.randn(
            len(input_batch),
            *problem.get_output_shape()
        )
