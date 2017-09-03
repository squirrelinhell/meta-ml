
from .Agent import Agent

from .BasicNet import BasicNet
from .Constant import Constant
from .Cycle import Cycle
from .Gauss import Gauss
from .GradAscent import GradAscent
from .RandomChoice import RandomChoice
from .Softmax import Softmax

import mandalka

@mandalka.node
class configure:
    def __init__(self, cls, **params):
        self.cls = cls
        self.params = params
        self.results = set()

    def __call__(self, *args):
        ret = self.cls(*args, **self.params)
        self.results.add(ret)
        return ret

Zero = configure(Constant, value=0.0)
One = configure(Constant, value=1.0)
