
import mandalka
import numpy as np
import tensorflow as tf

from .BaseTFPolicy import BaseTFPolicy

@mandalka.node
class PolicyNet(BaseTFPolicy):
    def _policy(self, o_batch, a_shape, params,
            hidden_layers=[128]):
        params.alloc(101770)

        def affine(x, out_dim):
            assert len(x.shape.as_list()) == 2
            in_dim = int(x.shape[1])
            w = params.next_normal((in_dim, out_dim))
            b = params.next_normal((out_dim,))
            x = tf.matmul(x, w) + b
            return x * (1.0 / np.sqrt(in_dim + 1))

        layer = tf.reshape(
            o_batch,
            (-1, np.prod(o_batch.shape.as_list()[1:]))
        )

        for dim in hidden_layers:
            layer = affine(layer, dim)
            layer = tf.nn.relu(layer)

        return affine(layer, np.prod(a_shape))
