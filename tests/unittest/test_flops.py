"""CPU formula checks and an independently differentiated GRU oracle."""
import itertools

import pytest
import torch
from torch import nn

from qmodel.flops import FlopsEstimate, FlopsRegistry, GradientRequest, estimate_flops


class CountedMatmul(torch.autograd.Function):
    """Count matrix products actually visited by an explicit autograd graph."""

    @staticmethod
    def forward(ctx, value, weight, counts):
        # Save operands and record forward multiply-adds.
        ctx.save_for_backward(value, weight)
        ctx.counts = counts
        ctx.cost = 2 * value.shape[0] * weight.numel()
        counts[0] += ctx.cost
        return value @ weight.T

    @staticmethod
    def backward(ctx, gradient):
        # Compute and count only requested operand derivatives.
        value, weight = ctx.saved_tensors
        value_grad = weight_grad = None
        if ctx.needs_input_grad[0]:
            ctx.counts[1] += ctx.cost
            value_grad = gradient @ weight
        if ctx.needs_input_grad[1]:
            ctx.counts[2] += ctx.cost
            weight_grad = gradient.T @ value
        return value_grad, weight_grad, None


def explicit_gru(module, value, initial, counts):
    # Expand each direction using the documented PyTorch GRU gate equations.
    sequence = list(value.unbind(0))
    finals = []
    directions = 2 if module.bidirectional else 1
    for layer in range(module.num_layers):
        outputs = []
        for direction in range(directions):
            suffix = f"_l{layer}" + ("_reverse" if direction else "")
            state = initial[layer * directions + direction]
            order = range(len(sequence)) if direction == 0 else reversed(range(len(sequence)))
            states = [None] * len(sequence)
            for time in order:
                gates_input = CountedMatmul.apply(sequence[time], getattr(module, "weight_ih" + suffix), counts)
                gates_hidden = CountedMatmul.apply(state, getattr(module, "weight_hh" + suffix), counts)
                if module.bias:
                    gates_input = gates_input + getattr(module, "bias_ih" + suffix)
                    gates_hidden = gates_hidden + getattr(module, "bias_hh" + suffix)
                reset_input, update_input, new_input = gates_input.chunk(3, dim=-1)
                reset_hidden, update_hidden, new_hidden = gates_hidden.chunk(3, dim=-1)
                reset = torch.sigmoid(reset_input + reset_hidden)
                update = torch.sigmoid(update_input + update_hidden)
                new = torch.tanh(new_input + reset * new_hidden)
                state = (1 - update) * new + update * state
                states[time] = state
            outputs.append(states)
            finals.append(state)
        sequence = [torch.cat([states[time] for states in outputs], dim=-1) for time in range(len(sequence))]
    return torch.stack(sequence), torch.stack(finals)


@pytest.mark.parametrize("input_grad,weight_grad,backward", itertools.product([False, True], repeat=3))
def test_linear_counts(input_grad, weight_grad, backward):
    # Check fixed arithmetic and independent gradient switches.
    module = nn.Linear(4, 5)
    module.weight.requires_grad_(weight_grad)
    result = estimate_flops(module, (2, 3, 4), GradientRequest(backward, input_grad))
    assert result == FlopsEstimate(240, 240 * (backward and input_grad), 240 * (backward and weight_grad))
    assert result.total_flops == sum((result.forward_flops, result.backward_input_flops, result.backward_weight_flops))
    assert estimate_flops(module, (4,), GradientRequest(False, False)).forward_flops == 40


@pytest.mark.parametrize("groups,padding,stride,dilation", [(2, 0, 1, 1), (4, 2, 2, 2), (1, "same", 1, 2), (2, "valid", 2, 1)])
@pytest.mark.parametrize("batched", [False, True])
def test_conv_geometry(groups, padding, stride, dilation, batched):
    # Obtain output geometry independently through a real convolution.
    module = nn.Conv1d(4, 8, 3, groups=groups, padding=padding, stride=stride, dilation=dilation)
    shape = (2, 4, 9) if batched else (4, 9)
    output = module(torch.randn(shape))
    expected = 2 * output.numel() * (4 // groups) * 3
    assert estimate_flops(module, shape, GradientRequest(True, True)) == FlopsEstimate(expected, expected, expected)


def test_conv_fixed_and_freezing():
    # Verify the documented fixed example and a frozen weight.
    module = nn.Conv1d(4, 6, 3, groups=2)
    assert estimate_flops(module, (2, 4, 7), GradientRequest(True, True)) == FlopsEstimate(720, 720, 720)
    module.weight.requires_grad_(False)
    assert estimate_flops(module, (2, 4, 7), GradientRequest(True, False)) == FlopsEstimate(720, 0, 0)


def test_gru_fixed():
    # Verify omitted h0 derivatives and optional external input gradients.
    module = nn.GRU(4, 5)
    assert estimate_flops(module, (3, 2, 4), GradientRequest(True, True)) == FlopsEstimate(1620, 1320, 1620)
    assert estimate_flops(module, (3, 2, 4), GradientRequest(True, False)) == FlopsEstimate(1620, 600, 1620)
    assert estimate_flops(module, (3, 2, 4), GradientRequest(False, True, True)) == FlopsEstimate(1620, 0, 0)


@pytest.mark.parametrize("layers,bidirectional,steps", [(1, False, 1), (1, True, 3), (2, False, 3), (2, True, 3)])
@pytest.mark.parametrize("input_grad,h0_grad", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("freeze", ["none", "all", "weights", "first_layer", "alternating"])
def test_gru_autograd_oracle(layers, bidirectional, steps, input_grad, h0_grad, freeze):
    # Prepare small double-precision tensors and independent parameter freezes.
    torch.manual_seed(7)
    module = nn.GRU(2, 3, num_layers=layers, bidirectional=bidirectional).double()
    for index, (name, parameter) in enumerate(module.named_parameters()):
        frozen = freeze == "all" or (freeze == "weights" and name.startswith("weight"))
        frozen |= freeze == "first_layer" and "_l0" in name
        frozen |= freeze == "alternating" and index % 3 != 0
        parameter.requires_grad_(not frozen)
    value = torch.randn(steps, 2, 2, dtype=torch.double, requires_grad=input_grad)
    initial = torch.randn(layers * (2 if bidirectional else 1), 2, 3, dtype=torch.double, requires_grad=h0_grad)
    counts = [0, 0, 0]
    actual, actual_hidden = explicit_gru(module, value, initial, counts)
    expected, expected_hidden = module(value, initial)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_hidden, expected_hidden)

    # Compare derivatives and measured matrix counts against the shape-only estimate.
    leaves = [tensor for tensor in [value, initial, *module.parameters()] if tensor.requires_grad]
    if leaves:
        gradients = torch.autograd.grad(actual.sum() + actual_hidden.sum(), leaves)
        reference = torch.autograd.grad(expected.sum() + expected_hidden.sum(), leaves)
        for gradient, reference_gradient in zip(gradients, reference):
            torch.testing.assert_close(gradient, reference_gradient)
    assert estimate_flops(module, tuple(value.shape), GradientRequest(True, input_grad, h0_grad)) == FlopsEstimate(*counts)


@pytest.mark.parametrize("batch_first,bias", itertools.product([False, True], repeat=2))
def test_gru_layouts(batch_first, bias):
    # Check layout normalization without changing sequence geometry.
    module = nn.GRU(4, 5, batch_first=batch_first, bias=bias)
    request = GradientRequest(True, True, True)
    shape = (2, 3, 4) if batch_first else (3, 2, 4)
    assert estimate_flops(module, shape, request) == FlopsEstimate(1620, 1620, 1620)
    assert estimate_flops(module, (3, 4), request) == FlopsEstimate(810, 810, 810)


def test_registry_boundaries():
    # Require explicit registration and refuse implicit container traversal.
    registry = FlopsRegistry()
    estimator = lambda module, shape, request: FlopsEstimate(1, 2, 3)
    registry.register(nn.Linear, estimator)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(nn.Linear, estimator)
    assert registry.estimate(nn.Linear(2, 3), (2,), GradientRequest(False, False)).total_flops == 6
    with pytest.raises(NotImplementedError):
        estimate_flops(nn.Sequential(nn.Linear(2, 3)), (2,), GradientRequest(False, False))
    with pytest.raises(NotImplementedError):
        registry.estimate(nn.GRU(2, 3), (1, 2), GradientRequest(False, False))
    packed = nn.utils.rnn.pack_sequence([torch.randn(2, 2)])
    with pytest.raises(TypeError, match="PackedSequence"):
        estimate_flops(nn.GRU(2, 3), packed, GradientRequest(False, False))


@pytest.mark.parametrize("module,shape", [(nn.Linear(2, 3), (3,)), (nn.Conv1d(2, 3, 5), (2, 1)), (nn.GRU(2, 3), (0, 1, 2))])
def test_invalid_geometry(module, shape):
    # Fail visibly on incompatible geometry.
    with pytest.raises(ValueError):
        estimate_flops(module, shape, GradientRequest(True, True))


def test_exact_type_and_explicit_grad_mode():
    # Require explicit subclass registration even when forward is inherited.
    class CustomLinear(nn.Linear):
        pass

    with pytest.raises(NotImplementedError):
        estimate_flops(CustomLinear(2, 3), (2,), GradientRequest(True, True))

    # Honor explicit gradient requests independently of eval and no_grad.
    module = nn.Linear(2, 3).eval()
    with torch.no_grad():
        result = estimate_flops(module, (2,), GradientRequest(True, True))
    assert result == FlopsEstimate(12, 12, 12)
