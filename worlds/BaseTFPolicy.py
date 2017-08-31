
from . import Episode
from .BasePolicy import BasePolicy

class BaseTFPolicy(BasePolicy):
    def _policy(self, o_batch, a_shape, params, **kwargs): # -> a_batch
        raise NotImplementedError("_policy")

    def __init__(self, world,
            batch_size=128,
            normalize_rewards=False,
            **kwargs):
        import tensorflow as tf
        import numpy as np

        super().__init__(world, batch_size, normalize_rewards)

        # Prepare parameters
        params = Parameters()
        n_params = 0
        self.get_observation_shape = (1,)
        self.get_action_shape = lambda: (n_params,)
        self.get_reward_shape = lambda: (n_params,)

        # Build policy graph
        o_batch = tf.placeholder(
            tf.float32,
            (None,) + world.o_shape
        )
        a_batch = tf.reshape(
            self._policy(o_batch, world.a_shape, params, **kwargs),
            (-1,) + world.a_shape
        )

        # Get parameter vector
        n_params = params.size
        params = params.get_tensor()

        # Backpropagation
        a_grad = tf.placeholder(tf.float32, a_batch.shape)
        intermediate_reward = tf.reduce_sum(tf.reduce_mean(
            tf.multiply(a_grad, a_batch),
            axis=0
        ))
        p_grad = tf.gradients(intermediate_reward, params)[0]

        # Operations to modify parameter vector
        init_op = tf.variables_initializer([params])
        add_value = tf.placeholder(params.dtype, params.shape)
        add_op = tf.assign_add(params, add_value)

        def start_episode(seed):
            tf.set_random_seed(seed)
            rng = np.random.RandomState(seed)

            sess = tf.Session()
            sess.run(init_op)

            ep_len = 0.0

            def step(action):
                nonlocal ep_len

                sess.run(
                    add_op,
                    feed_dict={add_value: action}
                )

                history, ep_len = self._test_policy(
                    rng.randint(2**32, size=batch_size),
                    lambda o: sess.run(
                        a_batch,
                        feed_dict={o_batch: o}
                    )
                )

                o, a, r = zip(*history)
                return sess.run(
                    p_grad,
                    feed_dict={o_batch: o, a_grad: r}
                )

            def solve(o):
                return sess.run(
                    a_batch,
                    feed_dict={o_batch: [o]}
                )[0]

            ep = Episode()
            ep.next_observation = lambda: [ep_len]
            ep.step = step
            ep.solve = solve
            return ep

        self.start_episode = start_episode

class Parameters:
    def __init__(self):
        import tensorflow as tf
        import numpy as np

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
