
import mandalka

from . import Agent
from worlds import World

@mandalka.node
class BasicNet(Agent):
    def __init__(self, world, seed, hidden_layers, params):
        import numpy as np

        # Wrap the original world with a meta world of parameters
        world = BasicNetParamsWorld(world, hidden_layers)

        # Create agent object (if needed)
        if not isinstance(params, Agent):
            params = params(world, seed)
            assert isinstance(params, Agent)

        # Get a single parameter vector for this instance
        _, (params,) = params.step([None], [None])
        params = np.asarray(params)
        params.setflags(write=False)
        assert params.shape == world.act_shape

        def step(sta_batch, obs_batch):
            # Use the world's TF session to run computations
            return sta_batch, world.action_batch(params, obs_batch)

        self.step = step
        self.get_parameters = lambda: params

@mandalka.node
class BasicNetParamsWorld(World):
    def __init__(self, world, hidden_layers):
        import tensorflow as tf
        import numpy as np

        assert world.rew_shape == world.act_shape

        params = Parameters()
        self.get_action_shape = lambda: (params.get_size(),)
        self.get_reward_shape = lambda: (params.get_size(),)

        obs_batch = tf.placeholder(
            tf.float32,
            (None,) + world.obs_shape
        )
        obs_size = np.prod(obs_batch.shape.as_list()[1:])
        act_size = np.prod(world.act_shape)

        def count_params():
            n_params = 0
            last_dim = obs_size
            for dim in hidden_layers:
                n_params += (last_dim + 1) * dim
                last_dim = dim
            n_params += (last_dim + 1) * act_size
            return n_params

        def affine(x, out_dim):
            assert len(x.shape.as_list()) == 2
            in_dim = int(x.shape[1])
            w = params.next_normal((in_dim, out_dim))
            b = params.next_normal((out_dim,))
            x = tf.matmul(x, w) + b
            return x * (1.0 / np.sqrt(in_dim + 1))

        def layers(inp_batch):
            layer = tf.reshape(obs_batch, (-1, obs_size))
            for dim in hidden_layers:
                layer = affine(layer, dim)
                layer = tf.nn.relu(layer)

            return affine(layer, act_size)

        params.alloc(count_params())
        act_batch = tf.reshape(
            layers(obs_batch),
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

        sess = tf.Session()

        def action_batch(params_value, feed_obs_batch):
            return sess.run(
                act_batch,
                feed_dict={
                    params.get_tensor(): params_value,
                    obs_batch: feed_obs_batch
                }
            )

        def trajectory(outer_agent, seed):
            assert isinstance(outer_agent, Agent)

            # Get parameters from the outer agent, and build
            # an agent to hold them
            inner_agent = BasicNet(
                world,
                None,
                hidden_layers=hidden_layers,
                params=outer_agent,
            )
            params_value = inner_agent.get_parameters()

            # IMPORTANT: Actions in experience list are assumed
            # to be generated from the inner agent.
            # So learning only works if the underlying world
            # is strictly on-policy.
            traj = world.trajectory(inner_agent, seed)
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
        self.action_batch = action_batch

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
