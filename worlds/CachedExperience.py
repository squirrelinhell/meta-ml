
import mandalka

from .base import World
from agents import Agent

@mandalka.node
class CachedExperience(World):
    def __init__(self, world, agents, seed,
            test_each=0, max_batch=16):
        seed = Agent.split_seed(seed)

        if not isinstance(agents, list):
            agents = [agents]

        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.rew_shape

        cache = []
        def trajectory_batch(agent, seed_batch):
            trajs = world.trajectory_batch(agent, seed_batch)
            for t in trajs:
                cache.append(t)
            return trajs
        self.trajectory_batch = trajectory_batch

        def test(agent):
            mandalka.evaluate(agent)
            todo = test_each
            while todo >= 1:
                batch = min(max_batch, todo)
                trajectory_batch(agent, [seed() for _ in range(batch)])
                todo -= batch

        for a in agents:
            test(Agent.build(a, self, seed()))

        def trajectory_batch(_, seed_batch):
            return [cache[s % len(cache)] for s in seed_batch]
        self.trajectory_batch = trajectory_batch
        self.num_trajectories = lambda: len(cache)

    def __len__(self):
        return self.num_trajectories()
