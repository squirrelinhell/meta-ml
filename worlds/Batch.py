
import mandalka

from . import World

@mandalka.node
class Batch(World):
    def __init__(self, world, batch_size):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.rew_shape

        assert batch_size >= 2

        def trajectory(agent, seed):
            rng = np.random.RandomState(seed)
            del seed

            # Generate seeds for the whole batch
            seed_batch = rng.randint(2**32, size=batch_size)

            # Get trajectories from the underlying world
            trajs = world.trajectory_batch(agent, seed_batch)
            assert isinstance(trajs, list)

            # Concatenate experience from all trajectories
            output_traj = []
            for t in trajs:
                output_traj += t

            return output_traj

        self.trajectory_batch = (
            lambda agent, seed_batch:
                [trajectory(agent, seed) for seed in seed_batch]
        )
