
import mandalka

from .base import World
from agents import Agent

@mandalka.node
class CachedExperience(World):
    def __init__(self, world, agents,
            test_each=0, max_batch=16):
        import numpy as np

        if not isinstance(agents, list):
            agents = [agents]

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.rew_shape

        cache = []
        def trajectories(agent, n):
            trajs = world.trajectories(agent, n)
            for t in trajs:
                cache.append(t)
            return trajs
        self.trajectories = trajectories

        def test(agent):
            mandalka.evaluate(agent)
            todo = test_each
            while todo >= 1:
                batch = min(max_batch, todo)
                trajectories(agent, n=batch)
                todo -= batch

        for i, a in enumerate(agents):
            test(Agent.build(a, self, i))

        rng = np.random.RandomState()
        def trajectories(_, n):
            n = int(n)
            assert n >= 1
            idx = rng.choice(len(cache), size=n)
            # TODO: this is unsafe, values could be modified outside
            return [cache[i] for i in idx]

        self.trajectories = trajectories
        self.num_trajectories = lambda: len(cache)

    def __len__(self):
        return self.num_trajectories()
