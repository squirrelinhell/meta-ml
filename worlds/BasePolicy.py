
import numpy as np
import tensorflow as tf

from . import World, Episode

class BasePolicy(World):
    def __init__(self,
            world,
            batch_size=128,
            normalize_rewards=False,
            **kwargs):

        # eval_batch(o_batch) -> a_batch
        # reward_batch(a_batch, a_batch, r_batch) -> r_batch

        def _run_episodes(seed_batch, eval_batch, reward_batch):
            # Start <batch_size> episodes in parallel
            eps = [world.start_episode(s) for s in seed_batch]
            is_active = [True] * len(eps)
            history = []
            rews = [[] for _ in eps]

            # Gather rewards until all episodes are finished
            while True:
                active_i = []
                o_batch = []
                for i in range(len(eps)):
                    if is_active[i]:
                        try:
                            o = eps[i].get_observation()
                            o_batch.append(o)
                            active_i.append(i)
                        except StopIteration:
                            is_active[i] = False

                if len(active_i) < 1:
                    break

                a_batch = eval_batch(o_batch)
                assert a_batch.shape == (len(active_i),) + world.a_shape

                for o, a, i in zip(o_batch, a_batch, active_i):
                    r = eps[i].step(a)
                    rews[i].append(r)
                    history.append((o, a, i))

            # Averaging rewards from each episode will in the end
            # work like summing, because they appear multiple times
            rews = np.array([np.mean(r, axis=0) for r in rews])
            assert rews.shape == (batch_size,) + world.r_shape

            # Normalize rewards if requested
            if normalize_rewards:
                rews -= rew.mean()
                stddev = rews.std()
                assert stddev > 0.00001
                rews /= stddev

            history = [(o, a, rews[i]) for o, a, i in history]
            return reward_batch(*zip(*history))

        self._run_episodes = _run_episodes
