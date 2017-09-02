
import mandalka

from . import World

@mandalka.node
class Policy(World):
    def __init__(self, world, batch_size):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: (1,)

        # Summing over trajectories needs to make sense
        assert world.rew_shape == (1,)

        # Normalization needs to make sense
        assert batch_size >= 2

        def trajectory(agent, seed):
            # Generate seeds for the whole batch
            rng = np.random.RandomState(seed)
            seed_batch = rng.randint(2**32, size=batch_size)

            # Get trajectories from the underlying world
            trajs = world.trajectory_batch(agent, seed_batch)

            # Sum rewards from each trajectory
            traj_rewards = []
            traj_lengths = []
            for traj in trajs:
                rews = np.array([r for o, a, r in traj])
                assert len(rews.shape) == 1
                assert len(rews) >= 1

                traj_rewards.append(np.sum(rews))
                traj_lengths.append(len(rews))

            # Normalize rewards (relative to others in this batch)
            assert len(traj_rewards) == batch_size
            traj_rewards -= np.mean(traj_rewards)
            stddev = np.std(traj_rewards)
            if stddev > 0.000001:
                traj_rewards /= stddev

            # Calculating mean reward of each trajectory will
            # in the end work like summing, because the values
            # appear multiple times in the output trajectory
            traj_rewards /= traj_lengths

            output_traj = []
            for i, traj in enumerate(trajs):
                for o, a, _ in traj:
                    output_traj.append((o, a, traj_rewards[i]))

            return output_traj

        self.trajectory = trajectory
        self.inner_agent = lambda a, s: world.inner_agent(a, s)
