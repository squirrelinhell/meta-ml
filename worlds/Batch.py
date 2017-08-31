
import sys
import mandalka

from . import World

@mandalka.node
class Batch(World):
    def __init__(self, world, batch_size):
        import numpy as np

        self.get_observation_shape = lambda: world.o_shape
        self.get_action_shape = lambda: world.a_shape
        self.get_reward_shape = lambda: world.r_shape

        assert batch_size >= 1
        warning_printed = False

        def after_episode(agent, seed):
            nonlocal warning_printed

            # Use native support for batches if possible
            try:
                return world.after_episode(
                    agent=agent,
                    seed=seed,
                    batch_size=batch_size
                )
            except TypeError:
                if not warning_printed:
                    warning_printed = True
                    sys.stderr.write(
                        "Warning: native batches not supported by %s\n"
                            % mandalka.describe(world)
                    )

            # Otherwise, run episodes sequentially
            # TODO: this could be implemented using threads
            rng = np.random.RandomState(seed)
            all_exp = []
            w = world
            for _ in range(batch_size):
                w, exp = w.after_episode(agent, rng.randint(2**32))
                all_exp += exp

            if w == world:
                return self, all_exp
            else:
                return Batch(w, batch_size), all_exp

        self.after_episode = after_episode
