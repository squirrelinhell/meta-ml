
import mandalka

from . import Agent

class StatelessAgent(Agent):
    def __init__(self, *args, get_action=None, get_actions=None):
        import numpy as np

        assert len(args) == 0

        if get_action is not None:
            def step(states, observations):
                assert len(states) == len(observations)
                ret = np.array([get_action(o) for o in observations])
                return states, ret

        elif get_actions is not None:
            def step(states, observations):
                assert len(states) == len(observations)
                return states, np.asarray(get_actions(observations))

        else:
            raise ValueError("Specify get_action() or get_actions()")

        self.step = step
