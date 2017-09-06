
import mandalka

from .base import World

@mandalka.node
class Accuracy(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: (1,)

        assert world.act_shape == world.rew_shape

        def process_exp(o, a, r):
            assert a.shape == r.shape
            answer = np.argmax(a)
            wanted = np.argmax(a + r)
            return o, a, [1.0] if answer == wanted else [0.0]

        def trajectories(agent, n):
            trajs = world.trajectories(agent, n)

            return [
                [process_exp(*e) for e in traj]
                for traj in trajs
            ]

        self.trajectories = trajectories
