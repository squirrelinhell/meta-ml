
import mandalka

from . import Agent, Constant, Gauss
from worlds import World

@Agent.builder
@mandalka.node
class GradAscent(Agent):
    def __init__(self, world, seed, log_lr, n_steps, init=Gauss):
        import numpy as np
        rng = np.random.RandomState(seed)
        del seed

        # Wrap the original world with a meta world of learning rates
        world = GradAscentLRWorld(world, n_steps, init)
        log_lr = Agent.build(log_lr, world, rng.randint(2**32))

        # Run a single trajectory on GradAscentLRWorld
        value_agent = world.value_agent(log_lr, rng.randint(2**32))
        self.step = lambda s, o: value_agent.step(s, o)

@mandalka.node
class GradAscentStep(Agent):
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

        def step(sta_batch, obs_batch):
            act_batch = [value] * len(obs_batch)
            return sta_batch, act_batch

        self.step = step
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

            value_agent = Agent.build(init, world, rng.randint(2**32))

            prev_reward = agent_reward(value_agent, rng.randint(2**32))
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
                    log_lr=round(log_lr[0] + baseline, 2),
                )

                new_reward = reward(value_agent.last_gradient())
                traj.append((
                    [prev_reward],
                    log_lr,
                    [new_reward - prev_reward]
                ))
                prev_reward = new_reward

            return traj, value_agent

        self.trajectory = lambda a, s: trajectory(a, s)[0]
        self.value_agent = lambda a, s: trajectory(a, s)[1]
