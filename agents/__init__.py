
class Agent:
    def action_batch(self, o_batch):
        raise NotImplementedError("action_batch")

from .RandomChoice import RandomChoice
from .Repeat import Repeat
from .SampleFromPolicy import SampleFromPolicy
from .Softmax import Softmax
