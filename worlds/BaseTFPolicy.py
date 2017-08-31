
import mandalka

from . import World
from agents import Agent

class BaseTFPolicy(World, Agent):
    def _policy(self, o_batch, a_shape, params, **kwargs): # -> a_batch
        raise NotImplementedError("_policy")

    def __init__(self, world, ep_len=1000, after=None, **kwargs):
        import tensorflow as tf
        import numpy as np

        assert world.r_shape == world.a_shape

        def build_graph():
            params = Parameters()

            # Build policy graph
            o_batch = tf.placeholder(
                tf.float32,
                (None,) + world.o_shape,
                name="o_batch"
            )
            a_batch = tf.reshape(
                self._policy(o_batch, world.a_shape, params, **kwargs),
                (-1,) + world.a_shape,
                name="a_batch"
            )

            # Backpropagation
            a_grad = tf.placeholder(
                tf.float32,
                a_batch.shape,
                name="a_grad"
            )
            intermediate_reward = tf.reduce_sum(tf.reduce_mean(
                tf.multiply(a_grad, a_batch),
                axis=0
            ))
            grad = tf.gradients(
                intermediate_reward,
                params.get_tensor()
            )[0]
            tf.identity(grad, name="params_grad")

        def build_session():
            graph = tf.Graph()
            with graph.as_default():
                build_graph()

            sess = tf.Session(graph=graph)
            tf.reset_default_graph()

            # Add easy access to variables by name
            sess.names = dict()
            for op in graph.get_operations():
                sess.names[op.name] = op
                for t in op.outputs:
                    sess.names[t.name.split(":")[0]] = t

            tf.Session.__getattr__ = (
                lambda self, name: self.names.get(name)
            )

            return sess

        if after is None:
            sess = build_session()
        else:
            sess = after[0]._get_session()

        params_value = None
        n_params = sess.params.shape[0].value
        self.get_action_shape = lambda: (n_params,)
        self.get_reward_shape = lambda: (n_params,)

        def step(meta_agent, seed):
            nonlocal params_value

            # Gather experience by running policy with current
            # parameters as an agent in the underlying world
            w2, exp = world.after_episode(self, seed)

            # Get average parameter gradient of this experience batch
            # (it's OK to ignore actions, because they are exactly
            # the same as generated from current parameters)
            assert len(exp) >= 1
            o, a, r = zip(*exp)
            params_grad = sess.run(
                sess.params_grad,
                feed_dict={
                    sess.params: params_value,
                    sess.o_batch: o,
                    sess.a_grad: r
                }
            )

            # Compute new policy parameters
            params_grad = params_grad.reshape((1, n_params))
            params_update = meta_agent.action_batch(params_grad)[0]
            params_value += params_update

        def action_batch(o):
            return sess.run(
                sess.a_batch,
                feed_dict={
                    sess.params: params_value,
                    sess.o_batch: o
                }
            )

        self.action_batch = action_batch

        def init_params():
            nonlocal params_value

            if after is None:
                params_value = np.random.randn(n_params)
            else:
                prev, agent, seed = after
                rng = np.random.RandomState(seed)
                params_value = prev._get_params_value()
                for _ in range(ep_len):
                    step(agent, rng.randint(2**32))

        init_params()

        def after_episode(agent, seed):
            return self.__class__(
                world,
                ep_len,
                after=(self, agent, seed),
                **kwargs
            ), []

        self.after_episode = after_episode
        self._get_params_value = lambda: params_value
        self._get_session = lambda: sess

class Parameters:
    def __init__(self):
        import tensorflow as tf
        import numpy as np

        tensor = None
        size = None
        pos = 0

        def alloc(n_params):
            nonlocal tensor, size
            assert tensor is None
            tensor = tf.placeholder(
                tf.float32,
                n_params,
                name="params"
            )
            size = n_params

        def next_normal(shape):
            nonlocal pos
            l = np.prod(shape)
            ret = tensor[pos:pos+l]
            pos += l
            return tf.reshape(ret, shape)

        def get_size():
            assert tensor is not None
            if pos != size:
                raise ValueError(
                    "Model used %d instead of %d parameters"
                        % (pos, size)
                )
            return size

        self.alloc = alloc
        self.next_normal = next_normal
        self.get_size = get_size
        self.get_tensor = lambda: (get_size(), tensor)[1]
