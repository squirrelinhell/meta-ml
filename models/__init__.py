
import mandalka

class Model:
    #def __init__(self, problem, seed, ...)
    #    pass

    def predict(self, input):
        raise NotImplementedError("predict")

@mandalka.node
class ModelFactory:
    def __init__(self, model_class, **params):
        self.model_class = model_class
        self.params = params

    def fit(self, problem, seed):
        return globals()[self.model_class](
            problem=problem,
            seed=seed,
            **self.params
        )

from .BasicNet import BasicNet
from .Random import Random
