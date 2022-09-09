#!/usr/bin/env pytest
import importlib
import json
import os
import unittest
from unittest.mock import patch

import torch

import torchdynamo
from torchdynamo.optimizations import backends
from torchdynamo.optimizations.analysis import has_mutation
from torchdynamo.optimizations.inference import offline_autotuner
from torchdynamo.optimizations.log_args import conv_args_analysis
from torchdynamo.optimizations.normalize import Inplacifier
from torchdynamo.optimizations.normalize import normalize
from torchdynamo.testing import same


def has_onnxruntime():
    try:
        importlib.import_module("onnxruntime")
        return True
    except ImportError:
        return False


def has_ipex():
    try:
        importlib.import_module("intel_extension_for_pytorch")
        return True
    except ImportError:
        return False


def has_functorch():
    try:
        importlib.import_module("functorch")
        return True
    except ImportError:
        return False


class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Bn_Relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TestOptimizations(torchdynamo.testing.TestCase):
    def test_inplacifier(self):
        gm = torch.fx.symbolic_trace(Seq())
        normalize(gm)
        Inplacifier(gm).inplacify()
        gm.recompile()
        code = gm.code.replace(" ", "")
        self.assertIn("inplace=True", code)
        self.assertIn("out=linear_1", code)

    def test_has_mutation(self):
        gm = torch.fx.symbolic_trace(Seq())
        self.assertFalse(has_mutation(gm, torch.rand([10, 10])))

        class Mutating(torch.nn.Module):
            def __init__(self):
                super(Mutating, self).__init__()

            def forward(self, arg):
                return arg.add_(1)

        gm = torch.fx.symbolic_trace(Mutating())
        self.assertTrue(has_mutation(gm, torch.rand([10, 1, 1, 1])))

    def test_example_inputs(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        with torchdynamo.optimize_assert(compiler_fn):
            r3 = fn(a, (b, c), d)

        self.assertIsNotNone(r1)
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

    @patch.object(torchdynamo.config, "fake_tensor_propagation", False)
    @unittest.skipIf(not has_functorch(), "requires functorch")
    def test_log_conv_args(self):
        model = Conv_Bn_Relu(3, 32, kernel_size=3, stride=1)
        model = model.to(memory_format=torch.channels_last)
        model = model.eval()
        input = torch.randn(8, 3, 64, 64).contiguous(memory_format=torch.channels_last)
        r1 = model(input)
        # check tmp/conv_args.json exists and has keys as arg names
        filename = "tmp/conv_args.json"
        if os.path.exists(filename):
            os.remove(filename)
        with torchdynamo.optimize(conv_args_analysis), torch.no_grad():
            r2 = model(input)
        self.assertTrue(same(r1, r2.float(), tol=0.1))
        self.assertTrue(os.path.exists(filename))
        with open(filename) as f:
            args_dict = json.load(f)
            self.assertIn("convolution_default", args_dict.keys())
            conv_args_dict = args_dict["convolution_default"]
            self.assertIn("input", conv_args_dict.keys())
            self.assertIn("weight", conv_args_dict.keys())
            self.assertIn("bias", conv_args_dict.keys())
            self.assertIn("stride", conv_args_dict.keys())
            self.assertIn("padding", conv_args_dict.keys())
            self.assertIn("dilation", conv_args_dict.keys())
            self.assertIn("transposed", conv_args_dict.keys())
            self.assertIn("output_padding", conv_args_dict.keys())
            self.assertIn("groups", conv_args_dict.keys())
        os.remove(filename)

    @unittest.skipIf(not has_onnxruntime(), "requires onnxruntime")
    def test_export(self):
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        with torchdynamo.optimize_assert(offline_autotuner):
            r2 = s(i)
        self.assertTrue(same(r1, r2))

    @unittest.skipIf(not has_ipex(), "requires ipex")
    def test_ipex_fp32(self):
        model = Conv_Bn_Relu(3, 32, kernel_size=3, stride=1)
        model = model.to(memory_format=torch.channels_last)
        model = model.eval()
        input = torch.randn(8, 3, 64, 64).contiguous(memory_format=torch.channels_last)
        r1 = model(input)
        with torchdynamo.optimize(backends.ipex_fp32), torch.no_grad():
            r2 = model(input)
        self.assertTrue(same(r1, r2))
        self.assertEqual(r2.dtype, torch.float32)

    @unittest.skipIf(not has_ipex(), "requires ipex")
    def test_ipex_bf16(self):
        model = Conv_Bn_Relu(3, 32, kernel_size=3, stride=1)
        model = model.to(memory_format=torch.channels_last)
        model = model.eval()
        input = torch.randn(8, 3, 64, 64).contiguous(memory_format=torch.channels_last)
        r1 = model(input)
        with torchdynamo.optimize(
            backends.ipex_bf16
        ), torch.no_grad(), torch.cpu.amp.autocast():
            r2 = model(input)
        self.assertTrue(same(r1, r2.float(), tol=0.1))
        self.assertEqual(r2.dtype, torch.bfloat16)
