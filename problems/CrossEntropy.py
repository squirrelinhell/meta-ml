
import numpy as np
import mandalka

from . import Problem, RewardEpisode

class CrossEntropy(Problem):
    def __init__(self, problem):
        self.get_input_shape = problem.get_input_shape
        self.get_output_shape = problem.get_output_shape
        self.start_episode = lambda: CE_Episode(problem.start_episode())

class CE_Episode(RewardEpisode):
    def __init__(self, episode):
        assert isinstance(episode, SupervisedEpisode)

        def next_reward(output):
            correct = np.abs(episode.next_output())
            correct /= np.sum(correct)

            output = np.maximum(0.00001, output)
            assert output.shape == correct.shape

            # Compute cross entropy
            output_sum = np.sum(output)
            reward = np.sum(correct * np.log(output / output_sum))
            grad = correct / output - 1.0 / output_sum

            return (reward, grad)

        self.next_input = episode.next_input
        self.next_reward = next_reward
