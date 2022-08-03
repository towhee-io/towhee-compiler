import collections
import dataclasses
import functools
import inspect
from typing import Dict
from typing import Sequence

from typeguard import typechecked

from .. import variables as vars
from ..bytecode_transformation import create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..source import AttrSource
from ..variables import Variable


class ConstDictVariable(Variable):
    def __init__(self, items, user_cls, **kwargs):
        super(ConstDictVariable, self).__init__(**kwargs)
        self.items = items
        self.user_cls = user_cls

    def as_proxy(self):
        return {k: v.as_proxy() for k, v in self.items.items()}

    def python_type(self):
        return self.user_cls

    def reconstruct(self, codegen):
        if len(self.items) == 0:
            return [create_instruction("BUILD_MAP", 0)]
        keys = tuple(self.items.keys())
        for key in keys:
            codegen(self.items[key])
        return [
            codegen.create_load_const(keys),
            create_instruction("BUILD_CONST_KEY_MAP", len(keys)),
        ]

    def getitem_const(self, arg: Variable):
        index = arg.as_python_constant()
        return self.items[index].trace(self, arg)

    @typechecked
    def call_method(
        self,
        tx,
        name: str,
        args: Sequence[Variable],
        kwargs: Dict[str, Variable],
    ) -> Variable:
        options = vars.propagate(self, args, kwargs.values())
        val = self.items

        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            return self.getitem_const(args[0])
        elif name == "items":
            assert not (args or kwargs)
            return vars.basetuple(
                [
                    vars.basetuple([vars.constant(k, **options), v], **options)
                    for k, v in val.items()
                ],
                **options,
            )
        elif name == "keys":
            assert not (args or kwargs)
            return vars.basetuple(
                [vars.constant(k, **options) for k in val.keys()],
                **options,
            )

        elif name == "values":
            assert not (args or kwargs)
            return vars.basetuple(list(val.values()), **options)
        elif name == "__len__":
            assert not (args or kwargs)
            return vars.constant(len(self.items), **options)
        elif (
            name == "__setitem__"
            and args
            and args[0].is_python_constant()
            and self.mutable_local
        ):
            assert not kwargs and len(args) == 2
            newval = collections.OrderedDict(val)
            newval[args[0].as_python_constant()] = args[1]
            return tx.replace_all(self, self.modifed(newval, **options))
        elif (
            name in ("pop", "get")
            and args
            and args[0].is_python_constant()
            and args[0].as_python_constant() not in self.items
            and len(args) == 2
        ):
            # missing item, return the default value
            return args[1].add_options(options)
        elif (
            name == "pop"
            and args
            and args[0].is_python_constant()
            and self.mutable_local
        ):
            newval = collections.OrderedDict(val)
            result = newval.pop(args[0].as_python_constant())
            tx.replace_all(self, self.modifed(newval, **options))
            return result.add_options(options)
        elif (
            name == "update"
            and args
            and isinstance(args[0], ConstDictVariable)
            and self.mutable_local
        ):
            newval = collections.OrderedDict(val)
            newval.update(args[0].items)
            result = self.modifed(newval, **options)
            return tx.replace_all(self, result)
        elif (
            name in ("get", "__getattr__")
            and args
            and args[0].is_python_constant()
            and args[0].as_python_constant() in self.items
        ):
            result = self.items[args[0].as_python_constant()]
            return result.add_options(options)
        elif name == "__contains__" and args and args[0].is_python_constant():
            return vars.constant(args[0].as_python_constant() in self.items, **options)
        else:
            return super().call_method(tx, name, args, kwargs)

    def modifed(self, items, **options):
        """a copy of self with different items"""
        return self.clone(items=items, **options)

    def unpack_var_sequence(self, tx):
        options = vars.propagate([self])
        result = [vars.constant(k, **options) for k in self.items.keys()]
        return result


class DataClassVariable(ConstDictVariable):
    """
    This is a bit of a hack to deal with
    transformers.file_utils.ModelOutput() from huggingface.

    ModelOutput causes trouble because it a a mix of a dataclass and a
    OrderedDict and it calls super() methods implemented in C.
    """

    # ModelOutput() excludes None, though generic datclasses don't
    include_none = False

    @staticmethod
    @functools.lru_cache(None)
    def _patch_once():
        from transformers.file_utils import ModelOutput

        for obj in ModelOutput.__dict__.values():
            if callable(obj):
                skip_code(obj.__code__)

    @staticmethod
    def is_matching_cls(cls):
        try:
            from transformers.file_utils import ModelOutput

            return issubclass(cls, ModelOutput)
        except ImportError:
            return False

    @classmethod
    def is_matching_object(cls, obj):
        return cls.is_matching_cls(type(obj))

    @classmethod
    def create(cls, user_cls, args, kwargs, options):
        DataClassVariable._patch_once()

        skip_code(user_cls.__init__.__code__)
        keys = [f.name for f in dataclasses.fields(user_cls)]
        bound = inspect.signature(user_cls).bind(*args, **kwargs)
        bound.apply_defaults()
        assert set(bound.arguments.keys()) == set(keys)
        items = collections.OrderedDict()
        for key in keys:
            val = bound.arguments[key]
            if isinstance(val, Variable):
                items[key] = val
            else:
                if cls.include_none:
                    assert vars.is_literal(val)
                    items[key] = vars.constant(val)
                else:
                    assert val is None, f"unexpected {val}"

        if len(items) == 1 and not isinstance(items[keys[0]], vars.TensorVariable):
            unimplemented("DataClassVariable iterator constructor")
            # TODO(jansel): implement unpacking logic in ModelOutput.__post_init__

        return cls(items, user_cls, **options)

    @classmethod
    def wrap(cls, builder, obj):
        user_cls = type(obj)
        keys = [f.name for f in dataclasses.fields(user_cls)]

        excluded = []
        items = collections.OrderedDict()
        for key in keys:
            # __init__ function of a dataclass might not have yet defined the key
            if hasattr(obj, key):
                val = getattr(obj, key)
                var = builder.__class__(
                    tx=builder.tx, source=AttrSource(builder.source, key)
                )(val)
                if val is not None or cls.include_none:
                    items[key] = var
                else:
                    excluded.append(var)
        return cls(items, user_cls, **vars.propagate(excluded, items.values()))

    def __init__(self, items, user_cls, **options):
        super(DataClassVariable, self).__init__(items, user_cls, **options)
        assert self.is_matching_cls(user_cls)

    def reconstruct(self, codegen):
        codegen.extend_output([codegen._create_load_const(self.user_cls)])
        result = list(super().reconstruct(codegen))
        assert result[-1].opname == "BUILD_CONST_KEY_MAP"
        result.append(create_instruction("CALL_FUNCTION_KW", result.pop().argval))
        return result

    @typechecked
    def call_method(
        self,
        tx,
        name,
        args: Sequence[Variable],
        kwargs: Dict[str, Variable],
    ) -> Variable:
        options = vars.propagate(self, args, kwargs.values())
        if name == "__post_init__":
            user_fn = vars.usermethod(self.user_cls.__post_init__, self)
            return user_fn.call_function(tx, [], {})
        elif name == "__getitem__":
            assert not kwargs and len(args) == 1
            index = args[0].as_python_constant()
            if isinstance(index, str):
                return self.items[index].add_options(options)
            else:
                return (
                    self.call_method(tx, "to_tuple", [], {})
                    .call_method(tx, "__getitem__", args, kwargs)
                    .add_options(options)
                )
        elif name == "to_tuple":
            assert not (args or kwargs)
            return vars.basetuple(list(self.items.values()), **options)
        elif name == "__setattr__":
            name = "__setitem__"
        return super(DataClassVariable, self).call_method(tx, name, args, kwargs)

    @typechecked
    def var_getattr(self, tx, name: str) -> Variable:
        if name in self.items:
            return self.call_method(tx, "__getitem__", [vars.constant(name)], {})
        elif not self.include_none:
            defaults = {f.name: f.default for f in dataclasses.fields(self.user_cls)}
            if name in defaults:
                assert vars.is_literal(defaults[name])
                return vars.constant(defaults[name]).trace(self)
        super(DataClassVariable, self).var_getattr(tx, name)
