
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
        o_space = free_envs[0].observation_space
        a_space = free_envs[0].action_space

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
                i = np.argmax(a)
                assert a[i] >= 0.99
                assert a[i] <= 1.01
                return i

        else:
            raise ValueError("Unsupported action space")

        def after_episode(agent, seed, batch_size=1):
            envs = [get_env() for _ in range(batch_size)]
            rewards = [0.0] * len(envs)
            ep_lengths = [0] * len(envs)
            exp = []
            active_count = batch_size

            obs = [process_obs(e.reset()) for e in envs]
            while active_count >= 1:
                # Ask the agent to process a batch of observations
                actions = agent.action_batch(np.array(obs))
                active_idx = [
                    i for i in range(len(envs)) if envs[i] is not None
                ]
                assert len(active_idx)

                # Do a step in each environment and gather observations
                obs = []
                for i, a in zip(active_idx, actions):
                    o, r, done, _ = envs[i].step(process_action(a))
                    exp.append((o, a, i))
                    rewards[i] += r
                    ep_lengths[i] += 1
                    if done:
                        return_env(envs[i])
                        envs[i] = None
                        active_count -= 1
                    else:
                        obs.append(process_obs(o))

            # Normalize rewards
            if normalize_rewards:
                assert batch_size >= 2
                rewards -= np.mean(rewards)
                stddev = np.std(rewards)
                if stddev > 0.000001:
                    rewards /= stddev

            # Calculating mean reward of each episode will in
            # the end work like summing, because the values
            # appear multiple times in the experience list
            rewards = [r / l for r, l in zip(rewards, ep_lengths)]

            return self, [(o, a, [rewards[i]]) for o, a, i in exp]

        self.after_episode = after_episode
