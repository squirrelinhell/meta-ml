
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
        return super().__getattr__(name)

class Episode:
    def next_input(self): # -> input [ndarray] or None
        raise NotImplementedError("next_input")

    def next_reward(self, output): # -> (reward, gradient)
        raise NotImplementedError("next_reward")

    def interrupt(self):
        pass

from .Gym import Gym
from .Mnist import Mnist
