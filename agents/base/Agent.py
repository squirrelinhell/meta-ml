
import mandalka

class Agent:
    # def __init__(self, world, seed, ...)

    def step(self, states, observations): # -> (states, actions)
        raise NotImplementedError

    # Static methods

    def build(agent, world, seed):
        if isinstance(agent, Agent):
            return agent
        elif isinstance(agent, AgentBuilder):
            agent = agent(world, seed)
            assert isinstance(agent, Agent)
            return agent
        else:
            from values import Value
            agent = Value.get_float(agent, world.act_shape)
            assert isinstance(agent, Agent)
            return agent

    def builder(agent_cls):
        name = agent_cls.__name__
        if not hasattr(Agent, "cls_by_name"):
            Agent.cls_by_name = {}
        assert not name in Agent.cls_by_name
        Agent.cls_by_name[name] = agent_cls
        return AgentBuilder(name)

@mandalka.node
class AgentBuilder:
    def __init__(self, cls_name, **params):
        assert cls_name in Agent.cls_by_name
        self.cls_name = cls_name
        self.params = params

    def __call__(self, *args, **more_params):
        all_params = self.params.copy()
        all_params.update(more_params)

        if len(args) == 0 and len(more_params) >= 1:
            # Just adding some parameters, not building yet
            return AgentBuilder(self.cls_name, **all_params)
        else:
            # Really call object constructor
            assert len(args) == 2
            from worlds import World
            assert isinstance(args[0], World)
            assert isinstance(args[1], int)
            cls = Agent.cls_by_name[self.cls_name]
            return cls(*args, **all_params)

    # For convenience only

    def __getattr__(self, name):
        raise ValueError("Object '" + self.cls_name
            + "' has not yet been constructed")

    def __add__(self, other):
        from .. import AddValue
        return AddValue(base=self, add=other)
