
import gym
import gym.spaces
import numpy as np
import mandalka
import threading

from . import Problem, Episode

@mandalka.node
class Gym(Problem):
    def __init__(self, env_name):
        envs = [gym.make(str(env_name))]
        lock = threading.Lock()
        o_space = envs[0].observation_space
        a_space = envs[0].action_space

        if isinstance(o_space, gym.spaces.Box):
            self.get_input_shape = lambda: o_space.shape
            def process_obs(obs):
                return obs
        elif isinstance(o_space, gym.spaces.Discrete):
            self.get_input_shape = lambda: (o_space.n,)
            def process_obs(obs):
                one_hot = np.zeros(o_space.n,)
                one_hot[obs] = 1.0
                return one_hot
        else:
            raise ValueError("Unsupported observation space")

        if isinstance(a_space, gym.spaces.Box):
            self.get_output_shape = lambda: a_space.shape
            def process_output(output):
                assert output.shape == a_space.shape
                return output
        elif isinstance(a_space, gym.spaces.Discrete):
            self.get_output_shape = lambda: (a_space.n,)
            def process_output(output):
                assert output.shape == (a_space.n,)
                return np.argmax(output)
        else:
            raise ValueError("Unsupported action space")

        def start_episode():
            with lock:
                if len(envs) < 1:
                    envs.append(gym.make(str(env_name)))
                return GymEpisode(
                    envs.pop(),
                    on_episode_end,
                    process_obs,
                    process_output
                )

        def on_episode_end(env):
            with lock:
                envs.append(env)

        self.start_episode = start_episode

class GymEpisode(Episode):
    def __init__(self, env, on_end, process_obs, process_output):
        obs = env.reset()

        def get_input_batch(batch_size=1):
            if env is None:
                return None
            assert batch_size == 1
            return np.array([process_obs(obs)])

        def get_reward_batch(output_batch):
            nonlocal env, obs
            if env is None:
                return None
            output_batch = np.asarray(output_batch)
            assert len(output_batch) == 1
            action = process_output(output_batch[0])
            obs, reward, done, _ = env.step(action)
            if done:
                on_end(env)
                env = None
            return np.array([reward])

        self.get_input_batch = get_input_batch
        self.get_reward_batch = get_reward_batch
