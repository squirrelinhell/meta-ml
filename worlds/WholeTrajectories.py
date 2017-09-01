
import sys
import mandalka

from . import World

@mandalka.node
class WholeTrajectories(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: (1,)

        # This is only sensible for scalar rewards
        assert world.rew_shape == (1,)

        def trajectory_batch(agent, seed_batch):
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

            # Normalize rewards relative to this batch
            assert len(traj_rewards) >= 2
            traj_rewards -= np.mean(traj_rewards)
            stddev = np.std(traj_rewards)
            if stddev > 0.000001:
                traj_rewards /= stddev

            # Calculating mean reward of each trajectory will
            # in the end work like summing, because the values
            # appear multiple times in experience lists
            traj_rewards /= traj_lengths

            return [
                [(o, a, traj_rewards[i]) for o, a, _ in traj]
                for i, traj in enumerate(trajs)
            ]

        self.trajectory_batch = trajectory_batch
        self.inner_agent = lambda a, s: world.inner_agent(a, s)
