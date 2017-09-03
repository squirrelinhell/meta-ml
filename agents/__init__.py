
from .Agent import Agent

from .BasicNet import BasicNet
from .Constant import Constant
from .Cycle import Cycle
from .Gauss import Gauss
from .GradAscent import GradAscent
from .LearnOn import LearnOn
from .RandomChoice import RandomChoice
from .Reinforce import Reinforce
from .Softmax import Softmax

from configure import configure

BasicNet = configure(BasicNet)
Constant = configure(Constant)
Cycle = configure(Cycle)
Gauss = configure(Gauss)
GradAscent = configure(GradAscent)
LearnOn = configure(LearnOn)
One = configure(Constant, value=1.0)
RandomChoice = configure(RandomChoice)
Reinforce = configure(Reinforce)
Softmax = configure(Softmax)
Zero = configure(Constant, value=0.0)
