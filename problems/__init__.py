
class Problem:
    def get_input_shape(self): # -> shape [tuple of int]
        raise NotImplementedError("get_input_shape")

    def get_output_shape(self): # -> shape [tuple of int]
        raise NotImplementedError("get_output_shape")

    def start_episode(self): # -> episode
        raise NotImplementedError("start_episode")

    def __getattr__(self, name):
        if name == "input_shape":
            return self.get_input_shape()
        if name == "output_shape":
            return self.get_output_shape()
        return object.__getattribute__(self, name)

class Episode:
    def next_input(self): # -> input [ndarray]
        raise NotImplementedError("next_input")

    def next_reward(self, output): # -> (reward, gradient)
        raise NotImplementedError("next_reward")

    def __next__(self):
        return self.next_input()

    def __iter__(self):
        return self

from .Accuracy import Accuracy
from .CrossEntropy import CrossEntropy
from .Gym import Gym
from .ParallelEpisodes import ParallelEpisodes
from .Reinforce import Reinforce
