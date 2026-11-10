"""Shape-only estimates of dense matrix FLOPs for individual modules."""
from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, Tuple, Type

from torch import nn
from torch.nn.utils.rnn import PackedSequence


@dataclass(frozen=True)
class GradientRequest:
    """Explicit differentiation request, independent of global grad mode."""

    backward: bool
    input_requires_grad: bool
    h0_requires_grad: bool = False


@dataclass(frozen=True)
class FlopsEstimate:
    """Matrix FLOPs; backward_input includes internal activation gradients."""

    forward_flops: int
    backward_input_flops: int
    backward_weight_flops: int

    @property
    def total_flops(self) -> int:
        # Sum all three disjoint categories.
        return self.forward_flops + self.backward_input_flops + self.backward_weight_flops


Estimator = Callable[[nn.Module, Tuple[int, ...], GradientRequest], FlopsEstimate]


class FlopsRegistry:
    """Exact-type dispatch without module traversal or inherited rules."""

    def __init__(self) -> None:
        # Own registrations independently of other registries.
        self.estimators: Dict[Type[nn.Module], Estimator] = {}

    def register(self, module_type: Type[nn.Module], estimator: Estimator) -> None:
        # Reject accidental rule replacement.
        if module_type in self.estimators:
            raise ValueError(f"FLOPs estimator already registered: {module_type}")
        self.estimators[module_type] = estimator

    def estimate(self, module: nn.Module, input_shape: Tuple[int, ...], request: GradientRequest) -> FlopsEstimate:
        # Accept only dense shape metadata and exact registered types.
        if isinstance(input_shape, PackedSequence) or not isinstance(input_shape, tuple):
            raise TypeError("input_shape must be a dense shape tuple; PackedSequence is unsupported")
        if type(module) not in self.estimators:
            raise NotImplementedError(f"No FLOPs estimator for {type(module)}")
        return self.estimators[type(module)](module, input_shape, request)


def estimate_linear(module: nn.Linear, shape: Tuple[int, ...], request: GradientRequest) -> FlopsEstimate:
    # Count the dense matrix product across all leading dimensions.
    if not shape or shape[-1] != module.in_features:
        raise ValueError("Linear input shape does not match in_features")
    forward = 2 * prod(shape[:-1]) * module.in_features * module.out_features
    return FlopsEstimate(
        forward, forward * (request.backward and request.input_requires_grad),
        forward * (request.backward and module.weight.requires_grad),
    )


def estimate_conv1d(module: nn.Conv1d, shape: Tuple[int, ...], request: GradientRequest) -> FlopsEstimate:
    # Resolve batched and unbatched convolution geometry.
    if len(shape) not in (2, 3) or shape[-2] != module.in_channels:
        raise ValueError("Conv1d expects (C,L) or (B,C,L) matching in_channels")
    batch = shape[0] if len(shape) == 3 else 1
    length, kernel, stride, dilation = shape[-1], module.kernel_size[0], module.stride[0], module.dilation[0]
    if module.padding == "same":
        output_length = length
    else:
        padding = 0 if module.padding == "valid" else module.padding[0]
        output_length = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if output_length <= 0:
        raise ValueError("Conv1d output length must be positive")

    # Use the same dense connection count for each requested derivative.
    forward = 2 * batch * output_length * module.out_channels * (module.in_channels // module.groups) * kernel
    return FlopsEstimate(
        forward, forward * (request.backward and request.input_requires_grad),
        forward * (request.backward and module.weight.requires_grad),
    )


def estimate_gru(module: nn.GRU, shape: Tuple[int, ...], request: GradientRequest) -> FlopsEstimate:
    # Resolve dense sequence geometry without executing tensor operations.
    if len(shape) not in (2, 3) or shape[-1] != module.input_size:
        raise ValueError("GRU expects dense (T,I), (T,B,I), or batch_first (B,T,I)")
    if len(shape) == 2:
        steps, batch = shape[0], 1
    elif module.batch_first:
        batch, steps = shape[:2]
    else:
        steps, batch = shape[:2]
    if steps <= 0:
        raise ValueError("GRU sequence length must be positive")
    directions, hidden = 2 if module.bidirectional else 1, module.hidden_size
    forward = sum(
        6 * batch * steps * directions * hidden * ((module.input_size if layer == 0 else directions * hidden) + hidden)
        for layer in range(module.num_layers)
    )
    if not request.backward:
        return FlopsEstimate(forward, 0, 0)

    # Track activation dependencies, including trainable biases.
    layers = []
    inputs = [request.input_requires_grad] * steps
    for layer in range(module.num_layers):
        records = []
        for direction in range(directions):
            suffix = f"_l{layer}" + ("_reverse" if direction else "")
            weight_input = getattr(module, "weight_ih" + suffix).requires_grad
            weight_hidden = getattr(module, "weight_hh" + suffix).requires_grad
            trainable = weight_input or weight_hidden
            if module.bias:
                trainable |= getattr(module, "bias_ih" + suffix).requires_grad
                trainable |= getattr(module, "bias_hh" + suffix).requires_grad
            order = list(range(steps)) if direction == 0 else list(reversed(range(steps)))
            previous = request.h0_requires_grad
            states, previous_states = [False] * steps, [False] * steps
            for time in order:
                previous_states[time] = previous
                previous = inputs[time] or previous or trainable
                states[time] = previous
            records.append((order, states, previous_states, weight_input, weight_hidden))
        layers.append((inputs, records))
        inputs = [any(record[1][time] for record in records) for time in range(steps)]

    # Propagate demand from output and every layer's returned final state.
    backward_input, backward_weight = 0, 0
    demand = [True] * steps
    for layer in reversed(range(module.num_layers)):
        inputs, records = layers[layer]
        input_cost = 6 * batch * hidden * (module.input_size if layer == 0 else directions * hidden)
        hidden_cost = 6 * batch * hidden * hidden
        lower_demand = [False] * steps
        for order, states, previous_states, weight_input, weight_hidden in records:
            state_demand = list(demand)
            state_demand[order[-1]] = True
            for position in reversed(range(steps)):
                time = order[position]
                if not state_demand[time] or not states[time]:
                    continue
                backward_weight += input_cost * weight_input + hidden_cost * weight_hidden
                if inputs[time]:
                    backward_input += input_cost
                    lower_demand[time] = True
                if previous_states[time]:
                    backward_input += hidden_cost
                    if position > 0:
                        state_demand[order[position - 1]] = True
        demand = lower_demand
    return FlopsEstimate(forward, backward_input, backward_weight)


registry = FlopsRegistry()
registry.register(nn.Linear, estimate_linear)
registry.register(nn.Conv1d, estimate_conv1d)
registry.register(nn.GRU, estimate_gru)


def estimate_flops(module: nn.Module, input_shape: Tuple[int, ...], gradient_request: GradientRequest) -> FlopsEstimate:
    # Dispatch one module without visiting children.
    return registry.estimate(module, input_shape, gradient_request)


__all__ = ["FlopsEstimate", "GradientRequest", "FlopsRegistry", "estimate_flops", "registry"]
