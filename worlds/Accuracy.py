
import mandalka

from . import World

@mandalka.node
class Accuracy(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.o_shape
        self.get_action_shape = lambda: world.a_shape
        self.get_reward_shape = lambda: (1,)

        assert world.a_shape == world.r_shape

        def process_exp(o, a, r):
            assert a.shape == r.shape
            answer = np.argmax(a)
            wanted = np.argmax(a + r)
            return o, a, [1.0] if answer == wanted else [0.0]

        def after_episode(agent, seed):
            w2, exp = world.after_episode(agent, seed)

            exp = [process_exp(*e) for e in exp]

            return self if w2 == world else Accuracy(w2), exp

        self.after_episode = after_episode
