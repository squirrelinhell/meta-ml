
import mandalka

from . import Agent
from worlds import World

@Agent.builder
@mandalka.node
class WholeTrajectories(Agent):
    def __init__(self, world, seed, agent):
        agent = Agent.build(agent, WholeTrajectoriesWorld(world), seed)
        self.step = lambda s, o: agent.step(s, o)

@mandalka.node
class WholeTrajectoriesWorld(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: (1,)

        # Summing over trajectories needs to make sense
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

            # Normalize rewards (relative to others in this batch)
            assert len(traj_rewards) >= 2
            traj_rewards -= np.mean(traj_rewards)
            stddev = np.std(traj_rewards)
            if stddev > 0.000001:
                traj_rewards /= stddev

            # Calculating mean reward of each trajectory will
            # in the end work like summing, because the values
            # appear multiple times in the output trajectory
            traj_rewards /= traj_lengths

            return [
                [(o, a, traj_rewards[i]) for o, a, _ in traj]
                for i, traj in enumerate(trajs)
            ]

        self.trajectory_batch = trajectory_batch
