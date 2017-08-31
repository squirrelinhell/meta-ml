
class World:
    def get_observation_shape(self): # -> shape [tuple of int] or None
        return None

    def get_action_shape(self): # -> shape [tuple of int]
        raise NotImplementedError("get_action_shape")

    def get_reward_shape(self): # -> shape [tuple of int]
        raise NotImplementedError("get_reward_shape")

    def after_episode(self, agent, seed): # -> (world, experience)
        # Iterating over experience should return tuples:
        # (observation, action, reward)
        raise NotImplementedError("start_episode")

    # For convenience only
    def __getattr__(self, name):
        if name in ("o_shape", "observation_shape"):
            return self.get_observation_shape()
        if name in ("a_shape", "action_shape"):
            return self.get_action_shape()
        if name in ("r_shape", "reward_shape"):
            return self.get_reward_shape()
        return object.__getattribute__(self, name)

from .Accuracy import Accuracy
from .Batch import Batch
from .Distribution import Distribution
from .Mnist import Mnist
from .PolicyNet import PolicyNet
