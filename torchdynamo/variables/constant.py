from typing import Dict
from typing import Sequence

from typeguard import typechecked

from .. import variables
from ..utils import istype
from .base import VariableTracker


class ConstantVariable(VariableTracker):
    _python_type_ = "self"
    _as_python_constant_ = "self"
    _as_proxy_ = "self"

    @property
    def items(self):
        """
        Need this when adding a BaseListVariable and a ConstantVariable together.
        Happens in detectron2.
        """
        return self.unpack_var_sequence(tx=None)

    def getitem_const(self, arg: VariableTracker):
        index = arg.as_python_constant()
        return ConstantVariable(self.value[index]).trace(self, arg)

    @staticmethod
    def is_literal(obj):
        if type(obj) in (int, float, bool, type(None), str):
            return True
        if type(obj) in (list, tuple, set, frozenset):
            return all(ConstantVariable.is_literal(x) for x in obj)
        return False

    def unpack_var_sequence(self, tx):
        try:
            options = variables.propagate([self])
            return [ConstantVariable(x, **options) for x in self.as_python_constant()]
        except TypeError:
            raise NotImplementedError()

    def const_getattr(self, tx, name):
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError()
        return member

    @typechecked
    def call_method(
        self,
        tx,
        name: str,
        args: Sequence[VariableTracker],
        kwargs: Dict[str, VariableTracker],
    ) -> "VariableTracker":
        options = variables.propagate(self, args, kwargs.values())

        if istype(self.value, tuple):
            # empty tuple constant etc
            return variables.basetuple(
                self.unpack_var_sequence(tx), source=self.source, **options
            ).call_method(tx, name, args, kwargs)

        if isinstance(self.value, str) and name in str.__dict__.keys():
            try:
                const_args = [a.as_python_constant() for a in args]
                const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}

                method = getattr(self.value, name)
                return ConstantVariable(method(*const_args, **const_kwargs), **options)
            except NotImplementedError:
                pass
        return super(ConstantVariable, self).call_method(tx, name, args, kwargs)


class EnumVariable(VariableTracker):
    _python_type_ = "self"
    _as_python_constant_ = "self"
    _as_proxy_ = "self"
