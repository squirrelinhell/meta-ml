
import os
import mandalka
import numpy as np

from . import World, Episode

class Table(World):
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

        class Ep(Episode):
            def __init__(self, seed):
                rng = np.random.RandomState(seed)
                i = rng.choice(len(inputs))
                def get_observation():
                    return inputs[i]
                def step(action):
                    nonlocal i
                    action = np.asarray(action)
                    assert action.shape == labels[0].shape
                    reward = labels[i] - action
                    i = rng.choice(len(inputs))
                    return reward
                self.get_observation = get_observation
                self.step = step

        self.get_observation_shape = lambda: inputs[0].shape
        self.get_action_shape = lambda: labels[0].shape
        self.get_reward_shape = lambda: labels[0].shape
        self.start_episode = Ep
