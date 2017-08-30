
import numpy as np
import tensorflow as tf

from . import World, Episode

class TFPolicy(World):
    def _policy(self, o_batch, a_shape, params, **kwargs): # -> a_batch
        raise NotImplementedError("_compute_policy")

    def __init__(self, world,
            batch_size=128,
            normalize_reward=False,
            **kwargs):
        # Prepare parameters
        params = Parameters()
        n_params = 0
        self.get_action_shape = lambda: n_params
        self.get_reward_shape = lambda: n_params

        # Build policy graph
        o_batch = tf.placeholder(
            tf.float32,
            (None,) + world.o_shape
        )
        a_batch = tf.reshape(
            self._policy(o_batch, world.a_shape, params, **kwargs),
            (-1,) + world.a_shape
        )

        # Fix parameter vector
        n_params = params.size
        params = params.get_tensor()

        # Backpropagation
        a_grad = tf.placeholder(tf.float32, a_batch.shape)
        intermediate_reward = tf.reduce_sum(tf.reduce_mean(
            tf.multiply(a_grad, a_batch),
            axis=0
        ))
        o_grad = tf.gradients(intermediate_reward, params)[0]

        # Operations to modify parameter vector
        init_op = tf.variables_initializer([params])
        add_value = tf.placeholder(params.dtype, params.shape)
        add_op = tf.assign_add(params, add_value)

        class Ep(Episode):
            def __init__(self, seed):
                tf.set_random_seed(seed)

                sess = tf.Session()
                sess.run(init_op)

                n_step = 0
                def step(action):
                    nonlocal n_step
                    n_step += 1

                    # Update parameters
                    sess.run(
                        add_op,
                        feed_dict={add_value: action}
                    )

                    # Start <batch_size> episodes in parallel
                    eps = [
                        world.start_episode((seed, n_step, i))
                            for i in range(batch_size)
                    ]
                    for e in eps:
                        e.is_done = False
                        e.rews = []

                    # Gather rewards until all episodes are finished
                    while not np.array([e.is_done for e in eps]).all():
                        inps = [e.get_observation() for e in eps]
                        outs = sess.run(
                            a_batch,
                            feed_dict={o_batch: inps}
                        )
                        for e, o in zip(eps, outs):
                            if e.is_done:
                                continue
                            try:
                                r = e.step(o)
                                e.rews.append(r)
                            except StopIteration:
                                e.is_done = True
                    rews = [np.sum(e.rews, axis=0) for e in eps]
                    rews = np.asarray(rews)
                    assert rews.shape == outs.shape

                    # Normalize rewards if requested
                    if normalize_reward:
                        rews -= rews.mean()
                        stddev = rews.std()
                        if stddev < 0.00001:
                            return np.zeros(outs.shape)
                        rews /= stddev

                    # Backpropagation
                    grad = sess.run(
                        o_grad,
                        feed_dict={
                            o_batch: inps,
                            a_grad: rews
                        }
                    )
                    return grad

                def solve(obs):
                    return sess.run(
                        a_batch,
                        feed_dict={o_batch: [obs]}
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
