
import mandalka

from .BaseTFPolicy import BaseTFPolicy

@mandalka.node
class PolicyNet(BaseTFPolicy):
    def _policy(self, o_batch, a_shape, params,
            hidden_layers=[128]):
        import numpy as np
        import tensorflow as tf

        o_size = np.prod(o_batch.shape.as_list()[1:])
        a_size = np.prod(a_shape)

        def count_params():
            n_params = 0
            last_dim = o_size
            for dim in hidden_layers:
                n_params += (last_dim + 1) * dim
                last_dim = dim
            n_params += (last_dim + 1) * a_size
            return n_params

        params.alloc(count_params())

        def affine(x, out_dim):
            assert len(x.shape.as_list()) == 2
            in_dim = int(x.shape[1])
            w = params.next_normal((in_dim, out_dim))
            b = params.next_normal((out_dim,))
            x = tf.matmul(x, w) + b
            return x * (1.0 / np.sqrt(in_dim + 1))

        layer = tf.reshape(o_batch, (-1, o_size))
        for dim in hidden_layers:
            layer = affine(layer, dim)
            layer = tf.nn.relu(layer)

        return affine(layer, a_size)
