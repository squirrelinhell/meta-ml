
def default(f):
    def error():
        raise NotImplementedError(
            "World should implement trajectory() or trajectory_batch()"
        )
    f.assert_not_default = error
    return f

class World:
    def get_observation_shape(self): # -> {tuple of int} or None
        return None

    def get_action_shape(self): # -> {tuple of int}
        raise NotImplementedError("get_action_shape")

    def get_reward_shape(self): # -> {tuple of int}
        raise NotImplementedError("get_reward_shape")

    @default
    def trajectory(self, agent, seed): # -> list of (obs, act, reward)
        try:
            self.trajectory_batch.assert_not_default()
        except AttributeError:
            pass

        return self.trajectory_batch(agent, [seed])[0]

    @default
    def trajectory_batch(self, agent, seed_batch): # -> list of traj.
        try:
            self.trajectory.assert_not_default()
        except AttributeError:
            pass

        return [self.trajectory(agent, s) for s in seed_batch]

    def inner_agent(self, agent, seed): # -> {Agent}
        raise NotImplementedError("inner_agent")

    # For convenience only
    def __getattr__(self, name):
        if name in ("obs_shape", "observation_shape"):
            return self.get_observation_shape()
        if name in ("act_shape", "action_shape"):
            return self.get_action_shape()
        if name in ("rew_shape", "reward_shape"):
            return self.get_reward_shape()
        return object.__getattribute__(self, name)
