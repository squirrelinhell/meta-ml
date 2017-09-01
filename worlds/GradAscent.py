
import mandalka

from . import World
from agents import Repeat

@mandalka.node
class GradAscent(World):
    def __init__(self, world, n_steps=100):
        import numpy as np

        self.get_observation_shape = lambda: (1,)
        self.get_action_shape = lambda: (1,)
        self.get_reward_shape = lambda: (1,)

        assert world.rew_shape == world.act_shape

        def reward(v):
            norm = np.sqrt(np.sum(np.square(v)))
            return -np.log10(max(0.0000000001, norm))

        def trajectory(agent, seed):
            # Start from Gaussian noise
            rng = np.random.RandomState(seed)
            value = rng.randn(*world.act_shape)

            def current_gradient():
                t = world.trajectory(
                    Repeat(value),
                    rng.randint(2**32)
                )
                return np.sum([r for o, a, r in t], axis=0)

            grad = current_gradient()
            baseline = reward(grad)
            traj = []

            for _ in range(n_steps):
                prev_reward = reward(grad)

                # Get (logarithmic) learning rate from the outer agent
                action = agent.action([prev_reward - baseline])
                action = np.asarray(action)
                assert action.shape == (1,)
                lr = np.power(10.0, baseline + action[0])

                # Update current guess
                value += lr * grad
                grad = current_gradient()

                traj.append((
                    [prev_reward - baseline],
                    action,
                    [reward(grad) - prev_reward]
                ))

            return traj, Repeat(value), rng.randint(2**32)

        self.trajectory = lambda a, s: trajectory(a, s)[0]
        self.inner_agent = lambda a, s: world.inner_agent(
            *trajectory(a, s)[1:]
        )
