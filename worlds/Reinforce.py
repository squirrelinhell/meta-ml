
import mandalka

from . import World
from agents import SampleFromPolicy

@mandalka.node
class Reinforce(World):
    def __init__(self, world):
        import numpy as np

        self.get_observation_shape = lambda: world.o_shape
        self.get_action_shape = lambda: world.a_shape
        self.get_reward_shape = lambda: world.a_shape

        assert world.r_shape == (1,)

        def after_episode(agent, seed):
            w2, exp = world.after_episode(SampleFromPolicy(agent), seed)

            o, real_a, r = zip(*exp)
            real_a = np.asarray(real_a)
            r = np.asarray(r).reshape(-1)
            assert r.shape == (len(real_a),)

            # Off-policy version of REINFORCE (!!!)
            agent_a = agent.action_batch(o)
            exp = list(zip(o, agent_a, ((real_a - agent_a).T * r.T).T))

            return self if w2 == world else Reinforce(w2), exp

        self.after_episode = after_episode
