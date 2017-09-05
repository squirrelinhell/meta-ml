
import mandalka

from .base import Agent, StatelessAgent
from worlds import World
from values import Value

@Agent.builder
@mandalka.node
class GradAscent(StatelessAgent):
    def __init__(self, world, seed, log_lr, n_steps, init):
        import numpy as np
        seed = Agent.split_seed(seed)

        init = Value.get_float(init, world.act_shape, seed())

        # Wrap the original world with a meta world of learning rates
        world = GradAscentLRWorld(world, n_steps, init)
        log_lr = Agent.build(log_lr, world, seed())

        # Run a single trajectory on GradAscentLRWorld
        value = world.final_value(log_lr, seed())
        value.setflags(write=False)
        super().__init__(get_action=lambda _: value)

@mandalka.node
class GradAscentStep(StatelessAgent):
    def __init__(self, world, seed, previous, log_lr):
        import numpy as np

        assert world.rew_shape == world.act_shape
        assert world.obs_shape is None

        _, (value,) = previous.step([None], [None])

        # Get gradient for current value
        t = world.trajectory(previous, seed)
        grad = np.sum([r for o, a, r in t], axis=0)
        grad.setflags(write=False)

        # Update value according to learning rate
        value = value + np.power(10.0, log_lr) * grad
        value.setflags(write=False)

        super().__init__(get_action=lambda _: value)
        self.last_gradient = lambda: grad

@mandalka.node
class GradAscentLRWorld(World):
    def __init__(self, world, n_steps, init):
        import numpy as np

        self.get_observation_shape = lambda: (1,)
        self.get_action_shape = lambda: (1,)
        self.get_reward_shape = lambda: (1,)

        def reward(v):
            norm = np.sqrt(np.sum(np.square(v)))
            return -np.log10(max(0.0000000001, norm))

        def agent_reward(value_agent, seed):
            t = world.trajectory(value_agent, seed)
            return reward(np.sum([r for o, a, r in t], axis=0))

        def trajectory(lr_agent, seed):
            rng = np.random.RandomState(seed)
            del seed

            prev_reward = agent_reward(init, rng.randint(2**32))
            value_agent = init
            baseline = prev_reward
            traj = []
            state = [None]

            for _ in range(n_steps):
                # Get learning rate from the outer agent
                state, (log_lr,) = lr_agent.step(
                    state,
                    [prev_reward - baseline]
                )
                log_lr = np.asarray(log_lr)
                assert log_lr.shape == (1,)

                # Update value and build a new agent that holds it
                value_agent = GradAscentStep(
                    world,
                    rng.randint(2**32),
                    previous=value_agent,
                    log_lr=round(float(log_lr[0] + baseline), 2),
                )

                new_reward = reward(value_agent.last_gradient())
                traj.append((
                    [prev_reward],
                    log_lr,
                    [new_reward - prev_reward]
                ))
                prev_reward = new_reward

            _, (value,) = value_agent.step([None], [None])
            return traj, value

        self.trajectory = lambda a, s: trajectory(a, s)[0]
        self.final_value = lambda a, s: trajectory(a, s)[1]
