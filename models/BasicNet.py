
import mandalka
import numpy as np
import tensorflow as tf

from .TFBackprop import TFBackprop

def affine(x, out_dim):
    assert len(x.shape.as_list()) == 2
    in_dim = int(x.shape[1])
    stddev = 1 / np.sqrt(in_dim + 1)
    w = tf.Variable(tf.truncated_normal(
        stddev = stddev,
        shape = [in_dim, out_dim],
        dtype = x.dtype
    ))
    b = tf.Variable(tf.truncated_normal(
        stddev = stddev,
        shape = [out_dim],
        dtype = x.dtype
    ))
    return tf.matmul(x, w) + b

@mandalka.node
class BasicNet(TFBackprop):
    def _predict_batch(self, input_batch, output_shape,
            hidden_layers=[128]):
        layer = tf.reshape(
            input_batch,
            (-1, np.prod(input_batch.shape.as_list()[1:]))
        )

        for dim in hidden_layers:
            layer = affine(layer, dim)
            layer = tf.nn.relu(layer)

        return affine(layer, np.prod(output_shape))
