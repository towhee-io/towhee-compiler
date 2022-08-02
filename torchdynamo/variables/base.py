import collections
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set

from .. import variables
from ..exc import unimplemented
from ..source import AttrSource
from ..source import Source
from ..utils import dict_values
from ..utils import istype
from ..utils import odict_values


class MutableLocal:
    """
    Marker used to indicate this (list, iter, etc) was constructed in
    local scope and can be mutated safely in analysis without leaking
    state.
    """

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class VariableTracker:
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.
    """

    # fields to leave unmodified in apply()
    _nonvar_fields = ["value"]

    # predefined flags
    _python_type_ = None
    _as_python_constant_ = None
    _as_proxy_ = None

    @staticmethod
    def propagate(*vars: List[List["VariableTracker"]]):
        """Combine the guards from many VariableTracker into **kwargs for a new instance"""
        guards = set()

        def visit(var):
            if type(var) in (list, tuple, dict_values, odict_values):
                map(visit, var)
            elif isinstance(var, variables.BaseListVariable):
                guards.update(var.guards)
                map(visit, var.items)
            elif isinstance(var, variables.ConstDictVariable):
                guards.update(var.guards)
                visit(var.items.values())
            else:
                assert isinstance(var, VariableTracker), typestr(var)
                guards.update(var.guards)

        visit(vars)
        return {
            "guards": guards,
        }

    def clone(self, **kwargs):
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    @classmethod
    def copy(cls, value):
        """Deeper (but not full) copy, leaving FX and user objects alone"""
        return cls.apply(lambda x: x, value)

    @classmethod
    def apply(
        cls, fn: Callable[["VariableTracker"], "VariableTracker"], value, cache=None
    ):
        """
        Walk this object and call fn on all the VariableTracker
        instances to produce a new VariableTracker with the results.
        """
        if cache is None:
            cache = dict()

        idx = id(value)
        if idx in cache:
            return cache[idx][0]

        if isinstance(value, VariableTracker):
            updated_dict = dict(value.__dict__)
            for key in updated_dict.keys():
                if key not in value._nonvar_fields:
                    updated_dict[key] = cls.apply(fn, updated_dict[key], cache)
            result = fn(value.clone(**updated_dict))
        elif istype(value, list):
            result = [cls.apply(fn, v, cache) for v in value]
        elif istype(value, tuple):
            result = tuple(cls.apply(fn, v, cache) for v in value)
        elif istype(value, collections.OrderedDict):
            result = collections.OrderedDict(
                cls.apply(fn, v, cache) for v in value.items()
            )
        elif istype(value, dict):
            result = {k: cls.apply(fn, v, cache) for k, v in list(value.items())}
        else:
            result = value

        # save `value` to keep it alive and ensure id() isn't reused
        cache[idx] = (result, value)
        return result

    def add_guard(self, guard):
        return self.clone(guards=set.union(self.guards, {guard}))

    def add_guards(self, guards):
        assert isinstance(guards, set)
        return self.clone(guards=set.union(self.guards, guards))

    def add_options(self, options, *more):
        if more:
            return self.add_options(options).add_options(*more)
        if isinstance(options, VariableTracker):
            return self.add_guards(options.guards)
        assert isinstance(options, dict)
        return self.add_guards(options.get("guards", set()))

    def trace(self, var, *more):
        if more:
            return self.trace(var).trace(*more)
        if isinstance(var, VariableTracker):
            return self.add_guards(var.guards)
        return self.add_guards(VariableTracker.propagate([var])["guards"])

    def __str__(self):
        blacklist = ["guards", "source", "mutable_local"]
        args = {k: v for k, v in self.__dict__.items() if k not in blacklist}
        return f"{self.__class__.__name__}({args})"

    def __repr__(self):
        return str(self)

    def python_type(self):
        if type(self._python_type_) is type:
            return self._python_type_
        if isinstance(self._python_type_, str):
            if self._python_type_ == "self":
                return type(self.value)
        raise NotImplementedError(f"{self} has no type")

    def as_python_constant(self):
        """For constants"""
        if isinstance(self._as_python_constant_, str):
            if self._as_python_constant_ == "self":
                return self.value
        raise NotImplementedError(f"{self} is not a constant")

    def is_python_constant(self):
        try:
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

    def can_create_guard(self):
        try:
            self.create_guard(None)
            return True
        except NotImplementedError:
            return False

    def create_guard(self, fn):
        if self.source:
            return self.source.create_guard(fn)
        raise NotImplementedError()

    def replace_guards(self, guards, *fns):
        name = self.source.name()
        new_guards = {g for g in (guards or []) if g.name != name}
        new_guards.update(self.source.create_guard(fn) for fn in fns)
        return new_guards

    def const_getattr(self, tx, name: str) -> Any:
        """getattr(self, name) returning a python constant"""
        raise NotImplementedError()

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        """getattr(self, name) returning a new variable"""
        options = VariableTracker.propagate(self)
        value = self.const_getattr(tx, name)
        if not variables.is_literal(value):
            raise NotImplementedError()
        if self.source:
            options["source"] = AttrSource(self.source, name)
        return variables.constant(value, **options)

    def is_proxy(self):
        try:
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    def as_proxy(self):
        if isinstance(self._as_proxy_, str):
            if self._as_proxy_ == "self":
                return self.value
        raise NotImplementedError(str(self))

    def reconstruct(self, codegen):
        raise NotImplementedError()

    def unpack_var_sequence(self, tx):
        raise NotImplementedError()

    def has_unpack_var_sequence(self, tx):
        try:
            self.unpack_var_sequence(tx)
            return True
        except NotImplementedError:
            return False

    def num_parameters(self):
        unimplemented(f"num_parameters: {self}")

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        unimplemented(f"hasattr: {self}")

    def call_function(
        self,
        tx,
        args: Sequence["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        unimplemented(f"call_function {type(self)}: {self} {args} {kwargs}")

    def call_method(
        self,
        tx,
        name: str,
        args: Sequence["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__len__":
            assert not (args or kwargs)
            assert self.has_unpack_var_sequence(tx)

            length = len(self.unpack_var_sequence(tx))
            return variables.constant(length, **variables.propagate(self))
        elif name == "__getattr__":
            assert len(args) == 1
            assert args[0].is_python_constant()
            assert not kwargs

            attr_name = args[0].as_python_constant()
            return self.var_getattr(tx, attr_name).trace(self, args[0])
        elif name == "__contains__":
            assert len(args) == 1
            assert args[0].is_python_constant()
            assert not kwargs

            search = args[0].as_python_constant()
            result = search in self.value
            return variables.constant(result).trace(self, args, kwargs)
        raise unimplemented(f"call_method {type(self)}: {self} {name} {args} {kwargs}")

    def __init__(
        self,
        value: Any = None,
        guards: Optional[Set] = None,
        source: Source = None,
        mutable_local: MutableLocal = None,
    ):
        super(VariableTracker, self).__init__()
        self.guards = guards or set()
        self.source = source
        self.mutable_local = mutable_local
        self.value = value


def typestr(*objs):
    if len(objs) == 1:
        (obj,) = objs
        if isinstance(obj, VariableTracker):
            return str(obj)
        else:
            return type(obj).__name__
    else:
        return " ".join(map(typestr, objs))
