
class Agent:
    # def __init__(self, world, seed, ...)

    def step(self, sta_batch, obs_batch): # -> (sta_batch, act_batch)
        raise NotImplementedError

    def build(agent, world, seed):
        if isinstance(agent, Agent):
            return agent
        elif callable(agent):
            agent = agent(world, seed)
            assert isinstance(agent, Agent)
            return agent
        else:
            from . import Constant
            return Constant(world, 0, value=agent)
