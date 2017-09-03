
import mandalka

from . import Agent

@mandalka.node
class LearnOn(Agent):
    def __init__(self, world, seed, learn_world, agent):
        agent = Agent.build(agent, learn_world(world), seed)
        self.step = lambda s, o: agent.step(s, o)
