
import gym
import gym.spaces
import numpy as np
import mandalka
import threading

from . import Problem, RewardEpisode

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
                return Episode(
                    envs.pop(),
                    return_env,
                    process_obs,
                    process_output
                )

        def return_env(env):
            with lock:
                envs.append(env)

        self.start_episode = start_episode

class Episode(RewardEpisode):
    def __init__(self, env, on_end, process_obs, process_output):
        obs = env.reset()

        def next_input():
            nonlocal obs

            if env is None:
                return None

            assert obs is not None, "Run next_reward() first"
            ret = process_obs(obs)

            obs = None
            return ret

        def next_reward(output):
            nonlocal obs

            action = process_output(np.asarray(output))

            obs, reward, done, _ = env.step(action)
            if done:
                interrupt()

            return (reward, None)

        def interrupt():
            nonlocal env, obs

            if env is not None:
                on_end(env)
                env, obs = None, None

        self.next_input = next_input
        self.next_reward = next_reward
        self.interrupt = interrupt
