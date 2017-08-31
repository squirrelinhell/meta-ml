
import mandalka

from . import World

@mandalka.node
class Batch(World):
    def __init__(self, world, batch_size=128):
        import numpy as np

        self.get_observation_shape = lambda: world.o_shape
        self.get_action_shape = lambda: world.a_shape
        self.get_reward_shape = lambda: world.r_shape

        def after_episode(agent, seed):
            rng = np.random.RandomState(seed)

            all_exp = []
            w = world

            # Todo: this should run in multiple threads
            for _ in range(batch_size):
                w, exp = w.after_episode(agent, rng.randint(2**32))
                all_exp += exp

            if w == world:
                return self, all_exp
            else:
                return Batch(w, batch_size), all_exp

        self.after_episode = after_episode
