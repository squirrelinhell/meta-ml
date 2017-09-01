
from . import Agent

class RandomChoice(Agent):
    def __init__(self, world):
        import numpy as np

        rng = np.random.RandomState()
        act_size = np.prod(world.act_shape)

        def action(obs):
            ans = np.zeros(act_size)
            ans[rng.choice(act_size)] = 1.0
            return ans.reshape(world.act_shape)

        self.action = action
