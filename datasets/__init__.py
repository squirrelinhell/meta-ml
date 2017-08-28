
import os
import mandalka
import numpy as np

class Dataset:
    def _build(self, **kwargs): # -> (inputs, labels)
        raise NotImplementedError("_build")

    def __init__(self, **kwargs):
        path = ("__cache__/" + self.__class__.__name__.lower()
            + "_" + mandalka.unique_id(self))

        if os.path.exists(path):
            inputs = np.load(path + "/inputs.npy")
            labels = np.load(path + "/labels.npy")
        else:
            os.makedirs(path + ".tmp", exist_ok=True)
            inputs, labels = self._build(**kwargs)
            assert len(inputs) == len(labels)
            np.save(path + ".tmp/inputs.npy", inputs)
            np.save(path + ".tmp/labels.npy", labels)
            os.rename(path + ".tmp", path)

        self.get_inputs = lambda: inputs.copy()
        self.get_labels = lambda: labels.copy()
        self.get_input_shape = lambda: inputs[0].shape
        self.get_label_shape = lambda: labels[0].shape

        def shuffle_all(*arrays):
            p = np.arange(len(arrays[0]))
            np.random.shuffle(p)
            return [a[p].reshape(a.shape) for a in arrays]

        self.get_shuffled = lambda: shuffle_all(inputs, labels)

    def __getattr__(self, name):
        if name == "inputs":
            return self.get_inputs()
        if name == "labels":
            return self.get_labels()
        if name == "input_shape":
            return self.get_input_shape()
        if name == "label_shape":
            return self.get_label_shape()
        return object.__getattribute__(self, name)

from .Mnist import Mnist
