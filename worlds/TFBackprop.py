
import numpy as np
import tensorflow as tf

from . import World, Episode

class TFBackprop(World):
    def _compute_batch(self, inp_batch, out_size, **kwargs):
        raise NotImplementedError("_compute_batch")

    def __init__(self, world,
            batch_size=128,
            normalize_reward=False,
            **kwargs):
        self.params = Parameters()
        self.get_action_shape = lambda: self.params.size
        self.get_reward_shape = lambda: self.params.size

        inp_batch = tf.placeholder(
            tf.float32,
            (None,) + world.o_shape
        )

        out_batch = tf.reshape(
            self._compute_batch(
                inp_batch,
                np.prod(world.a_shape),
                **kwargs
            ),
            (-1,) + world.a_shape
        )

        grad_end = tf.placeholder(tf.float32, out_batch.shape)
        intermediate_reward = tf.reduce_sum(tf.reduce_mean(
            tf.multiply(grad_end, out_batch),
            axis=0
        ))

        params = self.params.get_tensor()
        grad_start = tf.gradients(intermediate_reward, params)[0]
        init_op = tf.variables_initializer([params])
        add_value = tf.placeholder(params.dtype, params.shape)
        add_op = tf.assign_add(params, add_value)

        class Ep(Episode):
            def __init__(self, seed):
                tf.set_random_seed(seed)

                sess = tf.Session()
                sess.run(init_op)

                eps = [
                    world.start_episode((seed, i))
                        for i in range(batch_size)
                ]

                def step(action):
                    sess.run(
                        add_op,
                        feed_dict={add_value: action}
                    )
                    inps = [e.get_observation() for e in eps]
                    outs = sess.run(
                        out_batch,
                        feed_dict={inp_batch: inps}
                    )
                    rews = [e.step(o) for e, o in zip(eps, outs)]
                    rews = np.asarray(rews)
                    assert rews.shape == outs.shape
                    if normalize_reward:
                        rews -= rews.mean()
                        stddev = rews.std()
                        if stddev < 0.00001:
                            return np.zeros(outs.shape)
                        rews /= stddev
                    grad = sess.run(
                        grad_start,
                        feed_dict={
                            inp_batch: inps,
                            grad_end: rews
                        }
                    )
                    return grad

                def solve(obs):
                    return sess.run(
                        out_batch,
                        feed_dict={inp_batch: [obs]}
                    )[0]

                self.step = step
                self.solve = solve

        self.start_episode = Ep

class Parameters:
    def __init__(self):
        params = None
        param_pos = 0

        def alloc(n_params):
            nonlocal params
            assert params is None
            params = tf.Variable(tf.truncated_normal(
                stddev = 1.0,
                shape = (n_params,),
                dtype = tf.float32
            ))
            self.size = n_params

        def next_normal(shape):
            nonlocal param_pos
            size = np.prod(shape)
            ret = params[param_pos:param_pos+size]
            param_pos += size
            return tf.reshape(ret, shape)

        def get_tensor():
            assert params is not None
            assert param_pos == self.size, (
                "Expected %d parameters" % param_pos
            )
            return params

        self.alloc = alloc
        self.next_normal = next_normal
        self.get_tensor = get_tensor
