
import numpy as np
import mandalka

from . import Problem, Episode

@mandalka.node
class Accuracy(Problem):
    def __init__(self, dataset):
        self.get_input_shape = dataset.get_input_shape
        self.get_output_shape = dataset.get_label_shape
        self.start_episode = lambda: AC_Episode(
            *[t.__iter__() for t in dataset.get_shuffled()]
        )

class AC_Episode(Episode):
    def __init__(self, inputs, labels):
        def next_reward(output):
            correct = np.asarray(next(labels))
            output = np.asarray(output)
            assert output.shape == correct.shape

            correct = np.argmax(correct)
            output = np.argmax(output)

            score = 1.0 if correct == output else 0.0
            return (score, None)

        self.next_input = inputs.__next__
        self.next_reward = next_reward
