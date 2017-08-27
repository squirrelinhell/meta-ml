
class World:
    def get_input_shape(self):
        # returns a tuple of int
        raise NotImplementedError("get_input_shape")

    def get_output_shape(self):
        # returns a tuple of int
        raise NotImplementedError("get_output_shape")

    def start_episode(self):
        # returns Episode
        raise NotImplementedError("start_episode")

class Episode:
    def get_input_batch(self):
        # returns ndarray with shape (N,) + get_input_shape()
        raise NotImplementedError("get_input_batch")

    def get_reward_batch(self, output_batch):
        # returns ndarray with size == len(output_batch)
        raise NotImplementedError("get_reward_batch")

    def get_reward_gradient(self, output_batch):
        # returns ndarray with shape output_batch.shape[1:]
        raise NotImplementedError("get_reward_gradient")

from .Gym import Gym
