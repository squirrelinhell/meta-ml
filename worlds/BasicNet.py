
import mandalka

from .BaseTFModel import BaseTFModel

@mandalka.node
class BasicNet(BaseTFModel):
    def _action_batch(self,
            obs_batch, act_shape, params,
            hidden_layers):
        import numpy as np
        import tensorflow as tf

        obs_size = np.prod(obs_batch.shape.as_list()[1:])
        act_size = np.prod(act_shape)

        def count_params():
            n_params = 0
            last_dim = obs_size
            for dim in hidden_layers:
                n_params += (last_dim + 1) * dim
                last_dim = dim
            n_params += (last_dim + 1) * act_size
            return n_params

        params.alloc(count_params())

        def affine(x, out_dim):
            assert len(x.shape.as_list()) == 2
            in_dim = int(x.shape[1])
            w = params.next_normal((in_dim, out_dim))
            b = params.next_normal((out_dim,))
            x = tf.matmul(x, w) + b
            return x * (1.0 / np.sqrt(in_dim + 1))

        layer = tf.reshape(obs_batch, (-1, obs_size))
        for dim in hidden_layers:
            layer = affine(layer, dim)
            layer = tf.nn.relu(layer)

        return affine(layer, act_size)
