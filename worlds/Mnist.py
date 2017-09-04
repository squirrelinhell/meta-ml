
import mandalka

from . import World
from .BaseTable import BaseTable

@mandalka.node
class Mnist(BaseTable):
    def __init__(self, test=False):
        def build_table():
            import tensorflow.examples.tutorials.mnist as tf_mnist
            data = tf_mnist.input_data.read_data_sets(
                "/tmp/mnist-download",
                validation_size=0,
                one_hot=True
            )

            data = data.test if test else data.train
            return data.images.reshape((-1, 28, 28)), data.labels

        super().__init__(build_table)
