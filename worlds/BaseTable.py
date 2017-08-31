
import mandalka

from . import World

class BaseTable(World):
    def _build(self, **kwargs): # -> (inputs, labels)
        raise NotImplementedError("_build")

    def __init__(self, batch_size=128, **kwargs):
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
            assert batch_size >= 1

            # Avoid PRNG initialization in case of 1-element batch
            if batch_size == 1:
                idx = [seed % len(inputs)]
            else:
                rng = np.random.RandomState(seed)
                idx = rng.randint(len(inputs), size=batch_size)

            # Request predictions from the agent
            pred = np.asarray(agent.action_batch(inputs[idx]))
            assert pred.shape == (batch_size,) + labels[0].shape

            # If predictions are from softmax, cross entropy
            # gradient is equal to (label - prediction)
            exp = [
                (inputs[i], p, labels[i] - p)
                for i, p in zip(idx, pred)
            ]
            return self, exp

        self.after_episode = after_episode
