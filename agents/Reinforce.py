
import mandalka

from . import Agent, RandomChoice
from worlds import World

@Agent.builder
@mandalka.node
class Reinforce(Agent):
    def __init__(self, world, seed, policy):
        import numpy as np
        rng = np.random.RandomState(seed)
        del seed

        policy = Agent.build(
            policy,
            ReinforceWorld(world),
            rng.randint(2**32)
        )

        agent = RandomChoice(world, rng.randint(2**32), p=policy)
        self.step = lambda s, o: agent.step(s, o)

@mandalka.node
class ReinforceWorld(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.act_shape

        assert world.rew_shape == (1,)

        def process_traj(agent, traj):
            # Unpack trajectory to vertical arrays
            obs, real_action, rew = zip(*traj)
            real_action = np.asarray(real_action)
            rew = np.asarray(rew).reshape(-1)
            assert rew.shape == (len(real_action),)

            # (TODO: this breaks agent state, need to process
            # multiple trajectories)
            _, agent_action = agent.step([None] * len(obs), obs)
            agent_action = np.asarray(agent_action)
            assert agent_action.shape == real_action.shape

            # True off-policy version of REINFORCE (!!!)
            return list(zip(
                obs,
                agent_action,
                ((real_action - agent_action).T * rew.T).T
            ))

        def trajectory_batch(agent, seed_batch):
            rng = np.random.RandomState(seed_batch[0])
            seed_batch[0] = rng.randint(2**32)

            trajs = world.trajectory_batch(
                RandomChoice(world, rng.randint(2**32), p=agent),
                seed_batch
            )
            return [process_traj(agent, t) for t in trajs]

        self.trajectory_batch = trajectory_batch
