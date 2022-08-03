from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

import torch
from typeguard import typechecked

from .. import variables as vars
from ..bytecode_transformation import create_instruction
from ..exc import unimplemented
from ..source import GetItemSource
from ..utils import namedtuple_fields
from ..variables import Variable
from .base import MutableLocal
from .constant import ConstantVariable


class BaseListVariable(Variable):
    @staticmethod
    def cls_for(obj):
        return {
            iter: ListIteratorVariable,
            list: ListVariable,
            slice: SliceVariable,
            torch.Size: SizeVariable,
            tuple: TupleVariable,
        }[obj]

    @typechecked
    def __init__(self, items: List[Variable], **kwargs: Any):
        super(BaseListVariable, self).__init__(**kwargs)
        self.items: List[Variable] = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def as_python_constant(self):
        if (
            isinstance(self._as_python_constant_, str)
            and self._as_python_constant_ == "self"
        ):
            return super().as_python_constant()
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        return self.python_type()(self._as_proxy())

    def getitem_const(self, arg: Variable):
        index = arg.as_python_constant()
        if isinstance(index, slice):
            if self.source is not None:
                return self.clone(
                    items=self.items[index],
                    source=GetItemSource(self.source, index),
                    mutable_local=None,
                ).trace(arg, self)
            else:
                return self.clone(items=self.items[index], mutable_local=None).trace(
                    arg, self
                )
        else:
            assert isinstance(index, int)
            return self.items[index].trace(arg, self)

    def unpack_var_sequence(self, tx):
        return [x.trace(self) for x in self.items]

    @typechecked
    def call_method(
        self,
        tx,
        name: str,
        args: Sequence[Variable],
        kwargs: Dict[str, Variable],
    ) -> Variable:
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            return self.getitem_const(args[0])
        elif name == "__add__":
            assert not kwargs and len(args) == 1
            return type(self)(self.items + args[0].items).trace(self, args, kwargs)
        elif (
            name == "__contains__"
            and len(args) == 1
            and args[0].is_python_constant()
            and all(x.is_python_constant() for x in self.items)
        ):
            assert not kwargs
            search = args[0].as_python_constant()
            result = any(x.as_python_constant() == search for x in self.items)
            return vars.constant(result).trace(self, args, kwargs)

        return super(BaseListVariable, self).call_method(tx, name, args, kwargs)


class RangeVariable(BaseListVariable):
    _python_type_ = range
    _as_python_constant_ = "self"

    def __init__(self, value, items=None, guards=None, **kwargs):
        if items is None:
            items = [vars.constant(x, guards=guards) for x in value]
        super().__init__(items, guards=guards, **kwargs)
        self.value = value

    def reconstruct(self, codegen):
        assert "range" not in codegen.tx.f_globals
        range_fn = codegen.create_load_global("range", add=True)
        if self.value.step == 1:
            if self.value.start == 0:
                return [
                    range_fn,
                    codegen.create_load_const(self.value.stop),
                    create_instruction("CALL_FUNCTION", 1),
                ]
            return [
                range_fn,
                codegen.create_load_const(self.value.start),
                codegen.create_load_const(self.value.stop),
                create_instruction("CALL_FUNCTION", 2),
            ]
        return [
            range_fn,
            codegen.create_load_const(self.value.start),
            codegen.create_load_const(self.value.stop),
            codegen.create_load_const(self.value.step),
            create_instruction("CALL_FUNCTION", 3),
        ]


class ListVariable(BaseListVariable):
    _python_type_ = list

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_LIST", len(self.items))]

    @typechecked
    def call_method(
        self,
        tx,
        name: str,
        args: Sequence[Variable],
        kwargs: Dict[str, Variable],
    ) -> Variable:
        options = vars.propagate(self, args, kwargs.values())
        if name == "append" and self.mutable_local:
            assert not kwargs
            assert len(args) == 1
            tx.replace_all(
                self,
                ListVariable(self.items + args, **options),
            )
            return vars.constant(None)
        elif (
            name in ("extend", "__iadd__")
            and self.mutable_local
            and args
            and args[0].has_unpack_var_sequence(tx)
        ):
            assert not kwargs
            assert len(args) == 1
            return tx.replace_all(
                self,
                ListVariable(
                    list(self.items) + list(args[0].unpack_var_sequence(tx)),
                    **options,
                ),
            )
        elif name == "insert" and self.mutable_local:
            assert not kwargs
            idx, value = args
            items = list(self.items)
            items.insert(idx.as_python_constant(), value)
            return tx.replace_all(
                self,
                ListVariable(items, **options),
            )
        elif name == "pop" and self.mutable_local:
            assert not kwargs
            items = list(self.items)
            result = items.pop(*[a.as_python_constant() for a in args])
            tx.replace_all(
                self,
                ListVariable(items, **options),
            )
            return result
        elif name == "clear" and self.mutable_local:
            assert not kwargs and not args
            return tx.replace_all(
                self,
                ListVariable([], **options),
            )
        elif (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs
            key, value = args
            items = list(self.items)
            items[key.as_python_constant()] = value
            result = ListVariable(items, **options)
            return tx.replace_all(self, result)
        else:
            return super().call_method(tx, name, args, kwargs)


class TupleVariable(BaseListVariable):
    _python_type_ = tuple

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_TUPLE", len(self.items))]

    @typechecked
    def call_method(
        self,
        tx,
        name: str,
        args: Sequence[Variable],
        kwargs: Dict[str, Variable],
    ) -> Variable:
        if (
            name in ("__add__", "__iadd__")
            and len(args) == 1
            and isinstance(args[0], TupleVariable)
        ):
            assert not kwargs
            return TupleVariable(self.items + args[0].items).trace(self, args, kwargs)
        elif (
            name in ("__add__", "__iadd__")
            and len(args) == 1
            and isinstance(args[0], ConstantVariable)
        ):
            assert not kwargs
            return TupleVariable(
                self.items + list(args[0].unpack_var_sequence(self))
            ).trace(self, args, kwargs)
        return super().call_method(tx, name, args, kwargs)


class SizeVariable(TupleVariable):
    """torch.Size(...)"""

    _python_type_ = torch.Size

    def reconstruct(self, codegen):
        codegen.load_import_from("torch", "Size")
        codegen.foreach(self.items)
        build_torch_size = [
            create_instruction("BUILD_TUPLE", len(self.items)),
            create_instruction("CALL_FUNCTION", 1),
        ]
        return build_torch_size


class ShapeVariable(TupleVariable):
    """
    Represents tensor.shape(...) and helps differentiate between a constant
    TupleVariable and ShapeVariable.
    """

    pass


class NamedTupleVariable(TupleVariable):
    def __init__(self, items, tuple_cls, **kwargs):
        super().__init__(items, **kwargs)
        self.tuple_cls = tuple_cls

    def python_type(self):
        return self.tuple_cls

    def reconstruct(self, codegen):
        create_fn = getattr(self.tuple_cls, "_make", self.tuple_cls)
        codegen.append_output(codegen._create_load_const(create_fn))
        codegen.foreach(self.items)
        return [
            create_instruction("BUILD_TUPLE", len(self.items)),
            create_instruction("CALL_FUNCTION", 1),
        ]

    def var_getattr(self, tx, name):
        fields = namedtuple_fields(self.tuple_cls)
        if name not in fields:
            unimplemented(f"NamedTupleVariable.{name}")
        return self.items[fields.index(name)].trace(self)

    def call_hasattr(self, tx, name: str) -> Variable:
        options = vars.propagate(self)
        fields = namedtuple_fields(self.tuple_cls)
        return vars.constant(name in fields, **options)


class SliceVariable(BaseListVariable):
    _python_type_ = slice

    def __init__(self, items, **kwargs):
        start, stop, step = [vars.constant(None)] * 3
        if len(items) == 1:
            (stop,) = items
        elif len(items) == 2:
            start, stop = items
        elif len(items) == 3:
            start, stop, step = items
        else:
            assert False
        super().__init__([start, stop, step], **kwargs)

    def as_proxy(self):
        return slice(*self._as_proxy())

    def as_python_constant(self):
        return slice(*[x.as_python_constant() for x in self.items])

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_SLICE", len(self.items))]

    def var_getattr(self, tx, name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(f"slice.{name}")
        return self.items[fields.index(name)].trace(self)


class ListIteratorVariable(Variable):
    @typechecked
    def __init__(self, items, index: int = 0, **kwargs: Any):
        super(ListIteratorVariable, self).__init__(**kwargs)
        self.items = items
        self.index = index

    def next_variables(self):
        assert self.mutable_local
        if self.index >= len(self.items):
            raise StopIteration()
        return self.items[self.index].trace(self), ListIteratorVariable(
            self.items,
            self.index + 1,
            mutable_local=MutableLocal(),
            **vars.propagate([self]),
        )

    def as_python_constant(self):
        if self.index > 0:
            raise NotImplementedError()
        return iter([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        return [x.trace(self) for x in self.items[self.index :]]

    def reconstruct(self, codegen):
        remaining_items = self.items[self.index :]
        codegen.foreach(remaining_items)
        return [
            create_instruction("BUILD_TUPLE", len(remaining_items)),
            create_instruction("GET_ITER"),
        ]
