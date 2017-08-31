
import mandalka

from . import World

class BaseTable(World):
    def _build(self, **kwargs): # -> (inputs, labels)
        raise NotImplementedError("_build")

    def __init__(self, **kwargs):
        import os
        import numpy as np

        path = ("__cache__/" + self.__class__.__name__.lower()
            + "_" + mandalka.unique_id(self))

        if os.path.exists(path):
            inputs = np.load(path + "/inputs.npy", mmap_mode="r")
            labels = np.load(path + "/labels.npy", mmap_mode="r")
        else:
            os.makedirs(path + ".tmp", exist_ok=True)
            inputs, labels = self._build(**kwargs)
            assert len(inputs) == len(labels)
            np.save(path + ".tmp/inputs.npy", inputs)
            np.save(path + ".tmp/labels.npy", labels)
            os.rename(path + ".tmp", path)

        self.get_observation_shape = lambda: inputs[0].shape
        self.get_action_shape = lambda: labels[0].shape
        self.get_reward_shape = lambda: labels[0].shape

        def after_episode(agent, seed):
            i = seed % len(inputs)
            pred = np.asarray(agent.action_batch(inputs[i:i+1])[0])
            assert pred.shape == labels[i].shape
            return (self, [(inputs[i], pred, labels[i] - pred)])

        self.after_episode = after_episode
