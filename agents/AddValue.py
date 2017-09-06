
import mandalka

from .base import Agent, WrapperAgent
from worlds import World
from values import Value

@Agent.builder
@mandalka.node
class AddValue(WrapperAgent):
    def __init__(self, world, seed, base, add):
        add = Value.get_float(add, world.act_shape)
        base = Agent.build(base, AddValueWorld(world, add), seed)

        def process_action(a):
            return a + add.get()

        super().__init__(base, process_action=process_action)

@mandalka.node
class AddValueWorld(World):
    def __init__(self, world, add):
        self.get_observation_shape = lambda: world.obs_shape
        self.get_action_shape = lambda: world.act_shape
        self.get_reward_shape = lambda: world.rew_shape

        def trajectories(agent, n):
            assert isinstance(agent, Agent)
            return world.trajectories(
                AddValue(world, 0, base=agent, add=add),
                n
            )

        self.trajectories = trajectories
