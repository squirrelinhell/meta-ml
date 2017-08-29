
import numpy as np
import mandalka

from . import World, Episode

@mandalka.node
class Accuracy(World):
    def __init__(self, world):
        super().__init__(world)

        class Ep(Episode):
            def __init__(self, seed):
                ep = world.start_episode(seed)

                def step(action):
                    one_hot = action * 0.0
                    one_hot.flat[np.argmax(action)] = 1.0
                    reward = ep.step(one_hot)

                    if np.sum(np.abs(reward)) < 0.01:
                        return 1.0
                    else:
                        return 0.0

                self.get_observation = ep.get_observation
                self.step = step

        self.start_episode = Ep
