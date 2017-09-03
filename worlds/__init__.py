
from .World import World

from .Accuracy import Accuracy
from .Batch import Batch
from .Gym import Gym
from .Mnist import Mnist
from .WholeTrajectories import WholeTrajectories

from configure import configure

Batch = configure(Batch)
