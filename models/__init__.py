
import mandalka

class Model:
    #def __init__(self, world, seed, ...)
    #    pass

    def predict_batch(self, inputs):
        raise NotImplementedError("predict_batch")

@mandalka.node
class ModelFactory:
    def __init__(self, model_class, **params):
        self.model_class = model_class
        self.params = params

    def fit(self, world, seed):
        return globals()[self.model_class](
            world=world,
            seed=seed,
            **self.params
        )

from .Random import Random
