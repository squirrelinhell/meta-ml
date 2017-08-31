
class Agent:
    def action_batch(self, o_batch):
        raise NotImplementedError("action_batch")

from .AsDistribution import AsDistribution
from .Repeat import Repeat
