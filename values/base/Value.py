
import mandalka

class Value:
    # def __init__(self, shape, seed, ...)

    def get(self):
        raise NotImplementedError

    # Static methods

    def get_float(value, shape, seed):
        shape = tuple(int(s) for s in shape)
        if isinstance(value, ValueBuilder):
            value = value(shape, seed)
        elif isinstance(value, (float, list)):
            from .. import Constant
            value = Constant(shape, seed, value=value)
        from . import FloatValue
        assert isinstance(value, FloatValue)
        assert value.get().shape == shape
        return value

    def builder(value_cls):
        name = value_cls.__name__
        if not hasattr(Value, "cls_by_name"):
            Value.cls_by_name = {}
        assert not name in Value.cls_by_name
        Value.cls_by_name[name] = value_cls
        return ValueBuilder(name)

@mandalka.node
class ValueBuilder:
    def __init__(self, cls_name, **params):
        assert cls_name in Value.cls_by_name
        self.cls_name = cls_name
        self.params = params

    def __call__(self, *args, **more_params):
        all_params = self.params.copy()
        all_params.update(more_params)

        if len(args) == 0 and len(more_params) >= 1:
            # Just adding some parameters, not building yet
            return ValueBuilder(self.cls_name, **all_params)
        else:
            # Really call object constructor
            assert len(args) == 2
            args = (tuple(int(s) for s in args[0]), int(args[1]))
            cls = Value.cls_by_name[self.cls_name]
            return cls(*args, **all_params)

    def __getattr__(self, name):
        raise ValueError("Object '" + self.cls_name
            + "' has not yet been constructed")
