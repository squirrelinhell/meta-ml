
import os
import numpy as np
import mandalka

from . import Problem, SupervisedEpisode

class Table(Problem):
    def build(self, *args, **kwargs): # -> (inputs, outputs)
        raise NotImplementedError("build")

    def __init__(self, *args, **kwargs):
        path = ("__cache__/" + self.__class__.__name__.lower()
            + "_" + mandalka.unique_id(self))

        if os.path.exists(path):
            inputs = np.load(path + "/inputs.npy")
            outputs = np.load(path + "/outputs.npy")
        else:
            os.makedirs(path + ".tmp", exist_ok=True)
            inputs, outputs = self.build(*args, **kwargs)
            assert len(inputs) == len(outputs)
            np.save(path + ".tmp/inputs.npy", inputs)
            np.save(path + ".tmp/outputs.npy", outputs)
            os.rename(path + ".tmp", path)

        self.get_input_shape = lambda: inputs[0].shape
        self.get_output_shape = lambda: outputs[0].shape

        def shuffle_all(*arrays):
            p = np.arange(len(arrays[0]))
            np.random.shuffle(p)
            return [a[p].reshape(a.shape) for a in arrays]

        self.get_shuffled = lambda: shuffle_all(inputs, outputs)

    def start_episode(self):
        return TableEpisode(*self.get_shuffled())

class TableEpisode(SupervisedEpisode):
    def __init__(self, inputs, outputs):
        def next_input():
            nonlocal inputs
            if len(inputs) < 1:
                return None

            ret, inputs = inputs[0], inputs[1:]
            return ret

        def next_output():
            nonlocal outputs

            ret, outputs = outputs[0], outputs[1:]
            return ret

        self.next_input = next_input
        self.next_output = next_output
