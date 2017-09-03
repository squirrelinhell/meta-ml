
import mandalka

class Agent:
    # def __init__(self, world, seed, ...)

    def step(self, sta_batch, obs_batch): # -> (sta_batch, act_batch)
        raise NotImplementedError

@mandalka.node
class Configure:
    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs

    def __call__(self, *args):
        return self.cls(*args, **self.kwargs)

from .Cycle import Cycle
from .Gauss import Gauss
from .Softmax import Softmax
from .Zero import Zero
