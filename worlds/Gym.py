
import mandalka

from . import World

@mandalka.node
class Gym(World):
    def __init__(self, env_name, normalize_rewards=True):
        import gym
        import gym.spaces
        import numpy as np
        import threading

        free_envs = [gym.make(str(env_name))]
        lock = threading.Lock()

        def get_env():
            with lock:
                if len(free_envs) >= 1:
                    return free_envs.pop()

            return gym.make(str(env_name))

        def return_env(e):
            with lock:
                free_envs.append(e)

        self.get_reward_shape = lambda: (1,)
        obs_space = free_envs[0].observation_space
        act_space = free_envs[0].action_space

        if isinstance(obs_space, gym.spaces.Box):
            self.get_observation_shape = lambda: obs_space.shape
            def process_obs(o):
                return o

        elif isinstance(obs_space, gym.spaces.Discrete):
            self.get_observation_shape = lambda: (obs_space.n,)
            def process_obs(o):
                one_hot = np.zeros(obs_space.n,)
                one_hot[int(o)] = 1.0
                return one_hot

        else:
            raise ValueError("Unsupported observation space")

        if isinstance(act_space, gym.spaces.Box):
            self.get_action_shape = lambda: act_space.shape
            def process_action(a):
                assert a.shape == act_space.shape
                return a

        elif isinstance(act_space, gym.spaces.Discrete):
            self.get_action_shape = lambda: (act_space.n,)
            def process_action(a):
                assert a.shape == (act_space.n,)
                i = np.argmax(a)
                assert a[i] >= 0.99
                assert a[i] <= 1.01
                return i

        else:
            raise ValueError("Unsupported action space")

        def trajectory_batch(agent, seed_batch):
            envs = [get_env() for _ in seed_batch]
            trajs = [[] for _ in envs]

            obs = [process_obs(e.reset()) for e in envs]
            obs_idx = range(len(envs))

            while len(obs) >= 1:
                # Ask the outer agent to process observations
                actions = agent.action_batch(np.array(obs))
                actions = np.asarray(actions)
                assert len(actions) == len(obs)

                # Do a step in each environment and gather observations
                next_obs, next_obs_idx = [], []
                for o, a, i in zip(obs, actions, obs_idx):
                    next_o, r, done, _ = envs[i].step(process_action(a))
                    trajs[i].append((o, a, r))
                    if done:
                        return_env(envs[i])
                        envs[i] = None
                    else:
                        next_obs.append(process_obs(next_o))
                        next_obs_idx.append(i)

                obs, obs_idx = next_obs, next_obs_idx

            return trajs

        self.trajectory_batch = trajectory_batch
