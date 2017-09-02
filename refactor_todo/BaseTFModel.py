
import mandalka

from . import World
from agents import Agent

class BaseTFModel(World):
    def _action_batch(self, obs_batch, act_shape, params, **kwargs):
        raise NotImplementedError("_action_batch")

    def __init__(self, world, **kwargs):
        import tensorflow as tf
        import numpy as np

        assert world.rew_shape == world.act_shape

        # Build graph
        params = Parameters()
        obs_batch = tf.placeholder(
            tf.float32,
            (None,) + world.obs_shape
        )
        act_batch = tf.reshape(
            self._action_batch(
                obs_batch,
                world.act_shape,
                params,
                **kwargs
            ),
            (-1,) + world.act_shape
        )

        # Backpropagation
        act_grad = tf.placeholder(
            tf.float32,
            act_batch.shape
        )
        intermediate_reward = tf.reduce_sum(tf.reduce_mean(
            tf.multiply(act_grad, act_batch),
            axis=0
        ))
        params_grad = tf.gradients(
            intermediate_reward,
            params.get_tensor()
        )[0]

        # Use one TF session for everything
        sess = tf.Session()

        # Verify parameter count
        n_params = params.get_size()
        self.get_action_shape = lambda: (n_params,)
        self.get_reward_shape = lambda: (n_params,)

        class ParamsAgent(Agent):
            def __init__(self, params_value):
                def action_batch(feed_obs_batch):
                    return sess.run(
                        act_batch,
                        feed_dict={
                            params.get_tensor(): params_value,
                            obs_batch: feed_obs_batch
                        }
                    )
                self.action_batch = action_batch

        def trajectory(outer_agent, seed):
            # Get parameters from the outer agent
            params_value = outer_agent.action(None)
            params_agent = ParamsAgent(params_value)

            # IMPORTANT: Actions in experience list are assumed
            # to be the same as generated from ParamsAgent.
            # So learning only works if the underlying world
            # is strictly on-policy.
            traj = world.trajectory(params_agent, seed)
            all_obs, _, all_rew = zip(*traj)

            # Backpropagate gradient to parameters
            grad = sess.run(
                params_grad,
                feed_dict={
                    params.get_tensor(): params_value,
                    obs_batch: all_obs,
                    act_grad: all_rew
                }
            )

            return [(None, params_value, grad)]

        self.trajectory = trajectory
        self.inner_agent = lambda a, s: world.inner_agent(
            ParamsAgent(a.action(None)), s
        )

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
