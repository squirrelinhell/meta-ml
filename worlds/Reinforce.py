
import numpy as np
import mandalka

from . import World, Episode

@mandalka.node
class Reinforce(World):
    def __init__(self, world):
        super().__init__(world)

        class Ep(Episode):
            def __init__(self, seed):
                ep = world.start_episode((seed, 0))
                rng = np.random.RandomState((seed, 1))

                def step(action):
                    # Random draw from policy
                    flat = action.reshape((-1))
                    action_i = rng.choice(len(flat), p=flat)

                    # Send the choice to the environment
                    one_hot = action * 0.0
                    one_hot.flat[action_i] = 1.0
                    reward = ep.step(one_hot)

                    return (one_hot - action) * np.sum(reward)

                self.get_observation = ep.get_observation
                self.step = step

        self.start_episode = Ep
