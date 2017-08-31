
import mandalka

from . import World, Episode

@mandalka.node
class Reinforce(World):
    def __init__(self, world, test=False):
        super().__init__(world)
        if test:
            self.get_reward_shape = (1,)
        else:
            self.get_reward_shape = lambda: world.get_action_shape()

        self.start_episode = lambda seed: Ep(world, seed, test)

class Ep(Episode):
    def __init__(self, world, seed, test):
        import numpy as np

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
            reward = np.sum(ep.step(one_hot))

            if test:
                return [reward]
            else:
                # Cross entropy gradient
                return (one_hot - action) * reward

        self.next_observation = ep.next_observation
        self.step = step
