
class World:
    def get_observation_shape(self): # -> {tuple of int} or None
        return None

    def get_action_shape(self): # -> {tuple of int}
        raise NotImplementedError

    def get_reward_shape(self): # -> {tuple of int}
        raise NotImplementedError

    def trajectory(self, agent, seed): # -> list of (obs, act, reward)
        return self.trajectory_batch(agent, [seed])[0]

    def trajectory_batch(self, agent, seed_batch): # -> list of traj.
        raise NotImplementedError

    # For convenience only
    def __getattr__(self, name):
        if name in ("obs_shape", "observation_shape"):
            return self.get_observation_shape()
        if name in ("act_shape", "action_shape"):
            return self.get_action_shape()
        if name in ("rew_shape", "reward_shape"):
            return self.get_reward_shape()
        return object.__getattribute__(self, name)
