
import gym
import gym.spaces
import numpy as np
import mandalka
import threading

from . import World, Episode

@mandalka.node
class Gym(World):
    def __init__(self, env_name):
        envs = [gym.make(str(env_name))]
        lock = threading.Lock()

        self.get_reward_shape = lambda: (1,)
        o_space = envs[0].observation_space
        a_space = envs[0].action_space

        if isinstance(o_space, gym.spaces.Box):
            self.get_observation_shape = lambda: o_space.shape
            def process_obs(o):
                return o

        elif isinstance(o_space, gym.spaces.Discrete):
            self.get_observation_shape = lambda: (o_space.n,)
            def process_obs(o):
                one_hot = np.zeros(o_space.n,)
                one_hot[int(o)] = 1.0
                return one_hot

        else:
            raise ValueError("Unsupported observation space")

        if isinstance(a_space, gym.spaces.Box):
            self.get_action_shape = lambda: a_space.shape
            def process_action(a):
                assert a.shape == a_space.shape
                return a

        elif isinstance(a_space, gym.spaces.Discrete):
            self.get_action_shape = lambda: (a_space.n,)
            def process_action(a):
                assert a.shape == (a_space.n,)
                return np.argmax(a)

        else:
            raise ValueError("Unsupported action space")

        def start_episode(seed):
            with lock:
                if len(envs) < 1:
                    envs.append(gym.make(str(env_name)))
                return Ep(
                    envs.pop(),
                    return_env,
                    process_obs,
                    process_action
                )

        def return_env(env):
            with lock:
                envs.append(env)

        self.start_episode = start_episode

class Ep(Episode):
    def __init__(self, env, on_end, process_obs, process_action):
        obs = env.reset()

        def next_observation():
            nonlocal obs
            if env is None:
                raise StopIteration

            assert obs is not None, "Run step() first"
            ret = process_obs(obs)

            obs = None
            return ret

        def step(action):
            nonlocal env, obs
            if env is None:
                raise StopIteration

            action = np.asarray(action)
            action = process_action(action)

            obs, reward, done, _ = env.step(action)
            if done:
                on_end(env)
                env, obs = None, None

            return [reward]

        self.next_observation = next_observation
        self.step = step
