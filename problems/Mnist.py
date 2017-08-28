
import numpy as np
import mandalka
import tensorflow.examples.tutorials.mnist as tf_mnist

from . import Problem, Episode

@mandalka.node
class Mnist(Problem):
    def __init__(self):
        mnist = tf_mnist.input_data.read_data_sets(
            "__mnist__",
            validation_size=0,
            one_hot=True
        )

        self.get_input_shape = lambda: (28, 28)
        self.get_output_shape = lambda: (10,)

        def shuffle_all(*arrays):
            p = np.arange(len(arrays[0]))
            np.random.shuffle(p)
            return [a[p].reshape(a.shape) for a in arrays]

        inputs = mnist.train.images.reshape((-1, 28, 28))
        labels = mnist.train.labels
        self.start_episode = lambda: MnistEpisode(
            *shuffle_all(inputs, labels)
        )

class MnistEpisode(Episode):
    def __init__(self, all_inputs, all_labels):
        assert len(all_inputs) == len(all_labels)

        def next_input():
            nonlocal all_inputs

            ret, all_inputs = all_inputs[0], all_inputs[1:]
            return ret

        def next_reward(output):
            nonlocal all_labels
            correct, all_labels = all_labels[0], all_labels[1:]

            output = np.maximum(0.00001, output)
            assert output.shape == (10,)

            # Compute cross entropy
            output_sum = np.sum(output)
            reward = np.sum(correct * np.log(output / output_sum))
            grad = correct / output - 1.0 / output_sum

            return (reward, grad)

        self.next_input = next_input
        self.next_reward = next_reward
