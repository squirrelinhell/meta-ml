
import mandalka

from . import World
from agents import PolicyChoice

@mandalka.node
class Reinforce(World):
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

            # True off-policy version of REINFORCE (!!!)
            agent_action = agent.action_batch(obs)
            return list(zip(
                obs,
                agent_action,
                ((real_action - agent_action).T * rew.T).T
            ))

        def trajectory_batch(agent, seed_batch):
            trajs = world.trajectory_batch(
                PolicyChoice(agent),
                seed_batch
            )
            return [process_traj(agent, t) for t in trajs]

        self.trajectory_batch = trajectory_batch
        self.build_agent = lambda a: world.build_agent(PolicyChoice(a))
