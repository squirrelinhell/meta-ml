
import mandalka

from . import Agent
from worlds import World

@Agent.builder
@mandalka.node
class WholeTrajectories(Agent):
    def __init__(self, world, seed, agent, normalize=True):
        agent = Agent.build(
            agent,
            WholeTrajectoriesWorld(world, normalize),
            seed
        )
        self.step = lambda s, o: agent.step(s, o)

@mandalka.node
class WholeTrajectoriesWorld(World):
    def __init__(self, world, normalize):
        import os
        import sys
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: (1,)

        # Summing over trajectories needs to make sense
        assert world.rew_shape == (1,)

        def trajectories(agent, n):
            # Get trajectories from the underlying world
            trajs = world.trajectories(agent, n)

            # Sum rewards from each trajectory
            traj_rewards = np.zeros(len(trajs), dtype=np.float32)
            traj_lengths = np.zeros(len(trajs), dtype=np.int32)
            for i, traj in enumerate(trajs):
                rews = np.array([r for o, a, r in traj])
                assert len(rews.shape) == 2
                rews = rews.reshape(-1)
                assert len(rews) == len(traj)

                traj_rewards[i] = np.sum(rews)
                traj_lengths[i] = len(rews)

            if "DEBUG" in os.environ:
                sys.stderr.write(
                    "Mean reward/episode: %8.2f\n"
                        % np.mean(traj_rewards)
                )

            # Normalize rewards (relative to others in this batch)
            if normalize:
                assert len(traj_rewards) >= 2
                traj_rewards -= np.mean(traj_rewards)
                stddev = np.std(traj_rewards)
                if stddev > 0.000001:
                    traj_rewards /= stddev

            # Calculating mean reward of each trajectory will
            # in the end work like summing, because the values
            # appear multiple times in output
            traj_rewards /= traj_lengths

            return [
                [(o, a, traj_rewards[i]) for o, a, _ in traj]
                for i, traj in enumerate(trajs)
            ]

        self.trajectories = trajectories
