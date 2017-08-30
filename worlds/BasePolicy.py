
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
            is_active = np.array([True] * len(eps))
            reward_sum = np.zeros(self.r_shape)

            # Gather rewards until all episodes are finished
            while is_active.any():
                active = []
                inps = []
                for i in range(len(eps)):
                    if is_active[i]:
                        try:
                            inps.append(eps[i].get_observation())
                            active.append(i)
                        except StopIteration:
                            is_active[i] = False

                if len(active) < 1:
                    break

                outs = eval_batch(inps)
                rews = [eps[i].step(o) for i, o in zip(active, outs)]

                out_rews = reward_batch(inps, outs, rews)
                assert out_rews.shape == self.r_shape
                reward_sum += out_rews * len(inps)

            reward_sum /= batch_size

            # Normalize rewards if requested
            if normalize_rewards:
                reward_sum -= reward_sum.mean()
                stddev = reward_sum.std()
                if stddev < 0.00001:
                    return np.zeros(reward_sum.shape)
                reward_sum /= stddev

            return reward_sum

        self._run_episodes = _run_episodes
