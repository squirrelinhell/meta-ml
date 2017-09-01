
import mandalka

from . import World
from agents import Agent

class BaseTFPolicy(World):
    def _action_batch(self, obs_batch, act_shape, params, **kwargs):
        raise NotImplementedError("_policy")

    def __init__(self, world, batch_size=128, **kwargs):
        import tensorflow as tf
        import numpy as np

        assert world.rew_shape == world.act_shape

        # Build policy graph
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

        class InnerAgent(Agent):
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
            # Generate seeds for the whole batch
            rng = np.random.RandomState(seed)
            seed_batch = rng.randint(2**32, size=batch_size)

            # Get parameters from the outer agent
            params_value = outer_agent.action(None)

            # Gather experience by running policy with current
            # parameters as an agent in the underlying world
            inner_agent = InnerAgent(params_value)
            trajs = world.trajectory_batch(inner_agent, seed_batch)

            # Concatenate experience lists
            # IMPORTANT: We ignore actions, so this only works
            # if the underlying world is strictly on-policy
            all_obs = []
            all_rew = []
            for traj in trajs:
                for o, a, r in traj:
                    all_obs.append(o)
                    all_rew.append(r)

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
        self.get_policy = lambda a: InnerAgent(a.action(None))

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
