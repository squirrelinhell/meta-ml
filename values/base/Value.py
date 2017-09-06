
import mandalka

class Value:
    # def __init__(self, shape, ...)

    def get(self):
        raise NotImplementedError

    # Static methods

    def get_float(value, shape):
        shape = tuple(int(s) for s in shape)
        if isinstance(value, ValueBuilder):
            value = value(shape)
        elif isinstance(value, (float, list)):
            from .. import Constant
            value = Constant(shape, value=value)
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
            assert len(args) == 1
            shape = tuple(int(s) for s in args[0])
            cls = Value.cls_by_name[self.cls_name]
            return cls(shape, **all_params)

    # For convenience only

    def __getattr__(self, name):
        raise ValueError("Object '" + self.cls_name
            + "' has not yet been constructed")

    def __add__(self, other):
        from .. import Sum
        return Sum(value1=self, value2=other)

    def __mul__(self, other):
        from .. import Product
        return Product(value1=self, value2=other)
