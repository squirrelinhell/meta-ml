
import numpy as np
import mandalka

from . import Problem, RewardEpisode, SupervisedEpisode

@mandalka.node
class Accuracy(Problem):
    def __init__(self, problem):
        self.get_input_shape = problem.get_input_shape
        self.get_output_shape = problem.get_output_shape
        self.start_episode = lambda: Episode(problem.start_episode())

class Episode(RewardEpisode):
    def __init__(self, episode):
        assert isinstance(episode, SupervisedEpisode)

        def next_reward(output):
            correct = np.argmax(episode.next_output())
            output = np.argmax(output)

            score = 1.0 if correct == output else 0.0
            return (score, None)

        self.next_input = episode.next_input
        self.next_reward = next_reward
