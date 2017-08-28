
import mandalka
import numpy as np
import tensorflow as tf

from . import Model

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
class BasicNet(Model):
    def build(self, problem, inp,
            hidden_layers=[100, 100]):
        layer = tf.reshape(
            inp,
            (-1, np.prod(problem.input_shape))
        )

        for dim in hidden_layers:
            layer = affine(layer, dim)
            layer = tf.nn.relu(layer)

        layer = affine(layer, np.prod(problem.output_shape))

        return tf.reshape(
            layer,
            (-1,) + problem.output_shape
        )

    def __init__(self, problem, seed,
            episodes=10,
            batch_size=512,
            lr=0.01,
            eps=0.0001,
            **kwargs):
        inp = tf.placeholder(tf.float32, (None,) + problem.input_shape)
        out = self.build(problem, inp, **kwargs)

        # Backpropagation
        grad_in = tf.placeholder(tf.float32, (None,) + problem.output_shape)
        grad_ascend = tf.train.AdamOptimizer(
            learning_rate = lr,
            epsilon = eps
        ).minimize(-tf.reduce_mean(tf.multiply(grad_in, out), axis=0))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for _ in range(episodes):
            ep = problem.start_episode()
            ep_rewards = []
            while True:
                inputs = [ep.next_input() for _ in range(batch_size)]
                if inputs[-1] is None:
                    break

                outputs = sess.run(
                    out,
                    feed_dict={inp: inputs}
                )

                rewards, reward_grads = zip(*[
                    ep.next_reward(o) for o in outputs
                ])
                ep_rewards.append(np.mean(rewards))

                sess.run(
                    grad_ascend,
                    feed_dict={
                        inp: inputs,
                        grad_in: reward_grads
                    }
                )
            print("Mean episode reward: %.5f" % np.mean(ep_rewards))

        self.predict = lambda i: sess.run(
            out,
            feed_dict={inp: [i]}
        )[0]
