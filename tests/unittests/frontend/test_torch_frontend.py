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


def constant2(a, b):
    return a - b + (1.0 + 2)


class TestTorchFrontend:
    def test_bin_add(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_bin_add(x, y):
        ...         return x + y
        ...     result = test_bin_add(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %add : [#users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
            return (add,)
        """
        pass

    def test_const(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_const(x):
        ...         return x + 1.0
        ...     result = test_const(torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %add : [#users=1] = call_function[target=operator.add](args = (%x, 1.0), kwargs = {})
            return (add,)
        """
        pass

    def test_global(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_global(x, y):
        ...         return sub(x, y)
        ...     result = test_global(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %sub : [#users=1] = call_function[target=torch.sub](args = (%x, %y), kwargs = {})
            return (sub,)
        """
        pass

    def test_torch_function(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_torch_function(x, y):
        ...         return torch.sub(x, y)
        ...     result = test_torch_function(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %sub : [#users=1] = call_function[target=torch.sub](args = (%x, %y), kwargs = {})
            return (sub,)
        """
        pass

    def test_instance_method(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_instance_method(x, y):
        ...         return x.sub(y)
        ...     result = test_instance_method(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %sub : [#users=1] = call_method[target=sub](args = (%x, %y), kwargs = {})
            return (sub,)
        """
        pass

    def test_instance_method_with_local_var(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_instance_method_with_local_var(x, y):
        ...         target = x.sub
        ...         return target(y)
        ...     result = test_instance_method_with_local_var(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %sub : [#users=1] = call_method[target=sub](args = (%x, %y), kwargs = {})
            return (sub,)
        """
        pass

    def test_instance_method_with_args(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_instance_method_with_args(x, y):
        ...         target = x.sub
        ...         args = (y,)
        ...         return target(*args)
        ...     result = test_instance_method_with_args(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %sub : [#users=1] = call_method[target=sub](args = (%x, %y), kwargs = {})
            return (sub,)
        """
        pass

    def test_instance_method_with_args_kwargs(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_instance_method_with_args_kwargs(x, y):
        ...         target = torch.sub
        ...         args = (x, y,)
        ...         kwargs = {}
        ...         return target(*args, **kwargs)
        ...     result = test_instance_method_with_args_kwargs(torch.randn(10, 10), torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %sub : [#users=1] = call_function[target=torch.sub](args = (%x, %y), kwargs = {})
            return (sub,)
        """
        pass
    
    def test_function_call_nested(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_function_call_nested(x, y, z):
        ...         return constant2(x,y)+z
        ...     result = test_function_call_nested(
        ...                 torch.randn(10, 10),
        ...                 torch.randn(10, 10), 
        ...                 torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %z : torch.Tensor [#users=1] = placeholder[target=z]
            %sub : [#users=1] = call_function[target=operator.sub](args = (%x, %y), kwargs = {})
            %add : [#users=1] = call_function[target=operator.add](args = (%sub, 3.0), kwargs = {})
            %add_1 : [#users=1] = call_function[target=operator.add](args = (%add, %z), kwargs = {})
            return (add_1,)
        """
        pass

    def test_function_call_nested_v2(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_function_call_nested_v2(x, y, z):
        ...         return constant2(a=y,b=x)+z
        ...     result = test_function_call_nested_v2(
        ...                 torch.randn(10, 10),
        ...                 torch.randn(10, 10), 
        ...                 torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %z : torch.Tensor [#users=1] = placeholder[target=z]
            %sub : [#users=1] = call_function[target=operator.sub](args = (%y, %x), kwargs = {})
            %add : [#users=1] = call_function[target=operator.add](args = (%sub, 3.0), kwargs = {})
            %add_1 : [#users=1] = call_function[target=operator.add](args = (%add, %z), kwargs = {})
            return (add_1,)
        """
        pass
    
    def test_function_call_nested_v3(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_function_call_nested_v3(x, y, z):
        ...         return constant2(a=y,b=1.0)+z
        ...     result = test_function_call_nested_v3(
        ...                 torch.randn(10, 10),
        ...                 torch.randn(10, 10), 
        ...                 torch.randn(10, 10))
        graph():
            %y : torch.Tensor [#users=1] = placeholder[target=y]
            %z : torch.Tensor [#users=1] = placeholder[target=z]
            %sub : [#users=1] = call_function[target=operator.sub](args = (%y, 1.0), kwargs = {})
            %add : [#users=1] = call_function[target=operator.add](args = (%sub, 3.0), kwargs = {})
            %add_1 : [#users=1] = call_function[target=operator.add](args = (%add, %z), kwargs = {})
            return (add_1,)
        """
        pass

class TestTorchFrontendAdvanced:
    def test_device_constant(self):
        """
        >>> import torch
        >>> with compile(graph_compile_fn):
        ...     def test_device_constant(x, y, z):
        ...         return x + torch.ones(1, device=torch.device("cpu"))
        ...     result = test_device_constant(
        ...                 torch.randn(10, 10),
        ...                 torch.randn(10, 10), 
        ...                 torch.randn(10, 10))
        graph():
            %x : torch.Tensor [#users=1] = placeholder[target=x]
            %ones : [#users=1] = call_function[target=torch.ones](args = (1,), kwargs = {device: cpu})
            %add : [#users=1] = call_function[target=operator.add](args = (%x, %ones), kwargs = {})
            return (add,)
        """
        pass