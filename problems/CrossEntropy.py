
import numpy as np
import mandalka

from . import Problem, Episode

@mandalka.node
class CrossEntropy(Problem):
    def __init__(self, dataset):
        self.get_input_shape = dataset.get_input_shape
        self.get_output_shape = dataset.get_label_shape
        self.start_episode = lambda: CE_Episode(
            *[t.__iter__() for t in dataset.get_shuffled()]
        )

class CE_Episode(Episode):
    def __init__(self, inputs, labels):
        def next_reward(output):
            # Convert to probabilities
            output = np.asarray(output)
            output = np.exp(output - np.max(output))
            output /= output.sum()

            correct = np.asarray(next(labels))
            assert output.shape == correct.shape

            # Compute cross entropy
            reward = np.sum(
                correct * np.log(np.maximum(0.00001, output))
            )
            assert not np.isnan(reward)

            # Add gradient of this function
            grad = correct - output
            return (reward, grad)

        self.next_input = inputs.__next__
        self.next_reward = next_reward
