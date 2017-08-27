
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
    def __init__(self, inputs, labels):
        assert len(inputs) == len(labels)

        def get_input_batch(batch_size=128):
            nonlocal inputs
            if len(inputs) < batch_size:
                return None
            ret = inputs[0:batch_size]
            inputs = inputs[batch_size:]
            return ret

        def get_reward_gradient(output_batch):
            output_batch = np.abs(output_batch)

            nonlocal labels
            ok = labels[0:len(output_batch)]
            labels = labels[len(output_batch):]
            assert len(inputs) == len(labels)

            return np.mean(
                ok * (output_batch.T.sum(axis=0) / output_batch.T).T,
                axis=0
            )

        self.get_input_batch = get_input_batch
        self.get_reward_gradient = get_reward_gradient
