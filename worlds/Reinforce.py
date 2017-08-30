
import numpy as np
import mandalka

from . import World, Episode

@mandalka.node
class Reinforce(World):
    def __init__(self, world):
        super().__init__(world)
        self.start_episode = lambda seed: Ep(world, seed)

class Ep(Episode):
    def __init__(self, world, seed):
        rng = np.random.RandomState(seed)
        ep = world.start_episode(rng.randint(2**32))

        def step(action):
            # Safely calculate softmax
            action = np.asarray(action)
            action = np.exp(action - np.max(action))
            action /= action.sum()

            # Random draw from policy
            flat = action.reshape((-1))
            action_i = rng.choice(len(flat), p=flat)

            # Send the choice to the environment
            one_hot = action * 0.0
            one_hot.flat[action_i] = 1.0
            reward = ep.step(one_hot)

            # Cross entropy gradient
            return (one_hot - action) * np.sum(reward)

        self.get_observation = ep.get_observation
        self.step = step
