
import numpy as np
import mandalka

from . import Problem, RewardEpisode, SupervisedEpisode

@mandalka.node
class CrossEntropy(Problem):
    def __init__(self, problem):
        self.get_input_shape = problem.get_input_shape
        self.get_output_shape = problem.get_output_shape
        self.start_episode = lambda: Episode(problem.start_episode())

class Episode(RewardEpisode):
    def __init__(self, episode):
        assert isinstance(episode, SupervisedEpisode)

        def next_reward(output):
            correct = np.asarray(episode.next_output())
            assert (correct >= 0.0).all()
            correct /= correct.sum()

            # Linear outputs are assumed
            output = np.exp(output - np.max(output))
            assert output.shape == correct.shape

            # Compute cross entropy
            output_sum = output.sum()
            reward = np.sum(correct * np.log(
                np.maximum(0.00001, output / output_sum))
            )
            assert not np.isnan(output_sum)
            assert not np.isnan(reward)

            grad = correct - output / output_sum
            return (reward, grad)

        self.next_input = episode.next_input
        self.next_reward = next_reward
