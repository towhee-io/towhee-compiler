from typing import Dict
from typing import Sequence

from typeguard import typechecked

from .. import variables as vars
from ..utils import istype
from ..variables import Variable


class ConstantVariable(Variable):
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

    def getitem_const(self, arg: Variable):
        index = arg.as_python_constant()
        return ConstantVariable(self.value[index]).trace(self, arg)

    def unpack_var_sequence(self, tx):
        try:
            return [ConstantVariable(x).trace(self) for x in self.as_python_constant()]
        except TypeError:
            raise NotImplementedError()

    def const_getattr(self, tx, name: str):
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError()
        return member

    @typechecked
    def call_method(
        self,
        tx,
        name: str,
        args: Sequence[Variable],
        kwargs: Dict[str, Variable],
    ) -> Variable:
        if istype(self.value, tuple):
            # empty tuple constant etc
            return (
                vars.basetuple(self.items, source=self.source)
                .trace(self, args, kwargs)
                .call_method(tx, name, args, kwargs)
            )

        if isinstance(self.value, str) and name in str.__dict__.keys():
            try:
                const_args = [a.as_python_constant() for a in args]
                const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}

                retval = getattr(self.value, name)(*const_args, **const_kwargs)
                return ConstantVariable(retval).trace(self, args, kwargs)
            except NotImplementedError:
                pass
        return super(ConstantVariable, self).call_method(tx, name, args, kwargs)


class EnumVariable(Variable):
    _python_type_ = "self"
    _as_python_constant_ = "self"
    _as_proxy_ = "self"
