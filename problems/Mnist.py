
import numpy as np
import mandalka
import tensorflow.examples.tutorials.mnist as tf_mnist

from .Table import Table

@mandalka.node
class Mnist(Table):
    def build(self):
        data = tf_mnist.input_data.read_data_sets(
            "/tmp/mnist-download",
            validation_size=0,
            one_hot=True
        )

        return (
            data.train.images.reshape((-1, 28, 28)),
            data.train.labels
        )
