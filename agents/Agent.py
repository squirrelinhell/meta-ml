
def default(f):
    def error():
        raise NotImplementedError(
            "Agent should implement action() or action_batch()"
        )
    f.assert_not_default = error
    return f

class Agent:
    @default
    def action(self, obs): # -> {ndarray}
        try:
            self.action_batch.assert_not_default()
        except AttributeError:
            pass

        import numpy as np
        obs = np.asarray(obs)
        obs = obs.reshape((1,) + obs.shape)
        return self.action_batch(obs)[0]

    @default
    def action_batch(self, obs_batch): # -> {ndarray}
        try:
            self.action.assert_not_default()
        except AttributeError:
            pass

        import numpy as np
        if len(obs_batch) == 1:
            # Avoid creating a new array
            a = np.asarray(self.action(obs_batch[0]))
            return a.reshape((1,) + a.shape)
        else:
            actions = [self.action(o) for o in obs_batch]
            return np.array(actions)
