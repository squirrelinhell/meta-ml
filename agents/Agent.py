
class Agent:
    def action(self, obs): # -> {ndarray}
        assert not Agent.is_default(self.action_batch)
        import numpy as np

        obs = np.asarray(obs)
        obs = obs.reshape((1,) + obs.shape)
        return self.action_batch(obs)[0]

    def action_batch(self, obs_batch): # -> {ndarray}
        assert not Agent.is_default(self.action)
        import numpy as np

        if len(obs_batch) == 1:
            # Avoid creating a new array
            a = np.asarray(self.action(obs_batch[0]))
            return a.reshape((1,) + a.shape)
        else:
            actions = [self.action(o) for o in obs_batch]
            return np.array(actions)

    # Check if at least one of these is really implemented
    action.is_default = True
    action_batch.is_default = True
    def is_default(f):
        try:
            return f.is_default
        except AttributeError:
            return False
