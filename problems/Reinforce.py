
import mandalka
import numpy as np

from . import Problem, Episode

@mandalka.node
class Reinforce(Problem):
    def __init__(self, problem):
        self.get_input_shape = problem.get_input_shape
        self.get_output_shape = problem.get_output_shape
        self.start_episode = lambda: RE_Episode(problem.start_episode())

class RE_Episode(Episode):
    def __init__(self, episode):
        def next_reward(output):
            # Convert to probabilities
            output = np.asarray(output)
            output = np.exp(output - np.max(output))
            output /= output.sum()

            # Random draw from policy
            flat = output.reshape((-1))
            choice_i = np.random.choice(output.size, p=flat)
            choice_p = flat[choice_i]

            # Send the choice to the environment
            one_hot = output * 0.0
            one_hot.flat[choice_i] = 1.0
            reward, _ = episode.next_reward(one_hot)

            # Add gradient
            grad = one_hot - output
            return (reward * np.log(choice_p), grad)

        self.next_input = episode.next_input
        self.next_reward = next_reward
