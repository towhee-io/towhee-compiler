import torch
from torch import sub
from towhee.compiler.jit.hook import compile

from torchdynamo import config

config.raise_on_backend_error = False


def temp(*arg, **kws):
    return [1]


def graph_compile_fn(g, example_inputs=None):
    print(g.graph)
    return temp


class TestFrontendWithControl:
    def test_is_not_null(self):
        """
        >>> import torch
        
        graph when x and y are not None
        >>> with compile(graph_compile_fn):
        ...     def test_is_not_null(x, y):
        ...         if x is not None and y is not None:
        ...             return x + y
        ...     result = test_is_not_null(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %add : [#users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
            return (add,)
            
        null graph when there is None in arguments
        >>> with compile(graph_compile_fn):
        ...     def test_is_not_null(x, y):
        ...         if x is not None and y is not None:
        ...             return x + y
        ...     result = test_is_not_null(None, torch.randn(10, 10))
        """
        pass
    
    def test_inner_function(self):
        """
        >>> import torch
        
        >>> with compile(graph_compile_fn):
        ...     def test_inner_function(x, y):
        ...         def fn(x, y):
        ...             return torch.add(x, y)
        ...         return fn(x, y)
        ...     result = test_inner_function(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %add : [#users=1] = call_function[target=torch.add](args = (%x, %y), kwargs = {})
            return (add,)
        """
        pass
    
    def test_global_flag(self):
        """
        >>> import torch
        
        graph when flag is True
        >>> flag = True
        >>> with compile(graph_compile_fn):
        ...     def test_global_flag(x, y):
        ...         if flag:
        ...             return torch.add(x, y)
        ...         else:
        ...             return x
        ...     result = test_global_flag(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %add : [#users=1] = call_function[target=torch.add](args = (%x, %y), kwargs = {})
            return (add,)

        graph when x and y are not None
        >>> flag = False
        >>> with compile(graph_compile_fn):
        ...     def test_global_flag_2(x, y):
        ...         if flag:
        ...             return torch.add(x, y)
        ...         else:
        ...             return torch.sub(x, y)
        ...     result = test_global_flag_2(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %sub : [#users=1] = call_function[target=torch.sub](args = (%x, %y), kwargs = {})
            return (sub,)
        """
        pass
