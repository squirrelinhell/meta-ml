
import mandalka

from . import Agent

class WrapperAgent(Agent):
    def __init__(self, agent, *args, process_action=None):
        import numpy as np

        assert len(args) == 0

        def step(states, observations):
            assert len(states) == len(observations)
            states, actions = agent.step(states, observations)
            actions = np.array(actions)

            if process_action is not None:
                for i in range(len(actions)):
                    actions[i] = process_action(actions[i])

            return states, actions

        self.step = step
