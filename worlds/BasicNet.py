
import mandalka
import numpy as np
import tensorflow as tf

from .TFBackprop import TFBackprop

@mandalka.node
class BasicNet(TFBackprop):
    def _compute_batch(self, inp_batch, out_size, hidden_layers=[128]):
        self.params.alloc(101770)

        def affine(x, out_dim):
            assert len(x.shape.as_list()) == 2
            in_dim = int(x.shape[1])
            stddev = 1 / np.sqrt(in_dim + 1)
            w = self.params.next_normal((in_dim, out_dim)) / stddev
            b = self.params.next_normal((out_dim,)) / stddev
            return tf.matmul(x, w) + b

        layer = tf.reshape(
            inp_batch,
            (-1, np.prod(inp_batch.shape.as_list()[1:]))
        )

        for dim in hidden_layers:
            layer = affine(layer, dim)
            layer = tf.nn.relu(layer)

        return affine(layer, out_size)
