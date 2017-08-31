
import mandalka

from . import World, Episode

@mandalka.node
class Accuracy(World):
    def __init__(self, world):
        super().__init__(world)
        self.start_episode = lambda seed: Ep(world, seed)

class Ep(Episode):
    def __init__(self, world, seed):
        import numpy as np

        ep = world.start_episode(seed)

        def step(action):
            one_hot = action * 0.0
            one_hot.flat[np.argmax(action)] = 1.0
            reward = ep.step(one_hot)

            if np.sum(np.abs(reward)) < 0.01:
                return 1.0
            else:
                return 0.0

        self.next_observation = ep.next_observation
        self.step = step
