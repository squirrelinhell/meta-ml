
import numpy as np
import mandalka

from . import World, Episode

@mandalka.node
class Distribution(World):
    def __init__(self, world):
        super().__init__(world)

        class Ep(Episode):
            def __init__(self, seed):
                ep = world.start_episode(seed)

                def step(action):
                    # Safely calculate softmax
                    action = np.asarray(action)
                    action = np.exp(action - np.max(action))
                    action /= action.sum()

                    # Cross entropy gradient
                    return ep.step(action)

                self.get_observation = ep.get_observation
                self.step = step

        self.start_episode = Ep
