from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
from typing import Any

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
import torch.nn.functional as F

if hasattr(torch.backends, "nnpack"):
    torch.backends.nnpack.set_flags(False)


AXIS_NAMES = {
    2: ("row", "column"),
    3: ("depth", "row", "column"),
}

OPERATION_NAMES = {
    "conv": "convolution",
    "conv_transpose": "transposed convolution",
}

FUNCTION_NAMES = {
    ("conv", 2): "torch.nn.functional.conv2d",
    ("conv", 3): "torch.nn.functional.conv3d",
    ("conv_transpose", 2): "torch.nn.functional.conv_transpose2d",
    ("conv_transpose", 3): "torch.nn.functional.conv_transpose3d",
}


@dataclass(frozen=True)
class ConvolutionRequest:
    dimensions: int
    operation: str
    input_shape: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    input_channels: int
    output_channels: int
    groups: int
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    output_padding: tuple[int, ...]
    bias_enabled: bool
    input_values: list[Any]
    kernel_values: list[Any]
    bias_values: list[float] | None


def label_path(path: tuple[int, ...]) -> str:
    if not path:
        return "root"
    return "[" + "][".join(str(index) for index in path) + "]"


def _format_number(value: float) -> str:
    rounded = round(float(value), 6)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def _iter_indices(shape: tuple[int, ...]):
    return product(*(range(size) for size in shape))


def _ensure_numeric_scalar(value: Any, label: str, path: tuple[int, ...]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{label} contains a non-numeric value at {label_path(path)}: {value!r}"
        )
    return float(value)


def _validate_nested_values(
    values: Any, shape: tuple[int, ...], label: str, path: tuple[int, ...] = ()
) -> Any:
    if not shape:
        return _ensure_numeric_scalar(values, label, path)

    if not isinstance(values, list):
        raise ValueError(
            f"{label} must be a nested list matching shape {shape}. "
            f"Missing list at {label_path(path)}."
        )

    expected = shape[0]
    if len(values) != expected:
        raise ValueError(
            f"{label} expected length {expected} at {label_path(path)}, "
            f"but received {len(values)}."
        )

    return [
        _validate_nested_values(item, shape[1:], label, (*path, index))
        for index, item in enumerate(values)
    ]


def _validate_bool(name: str, value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def _validate_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _validate_spatial_shape(name: str, shape: Any, dimensions: int) -> tuple[int, ...]:
    if not isinstance(shape, list) or len(shape) != dimensions:
        raise ValueError(f"{name} must contain exactly {dimensions} dimensions.")

    normalized: list[int] = []
    for axis_name, axis_value in zip(AXIS_NAMES[dimensions], shape):
        if not isinstance(axis_value, int):
            raise ValueError(f"{name} axis '{axis_name}' must be an integer.")
        if axis_value < 2:
            raise ValueError(f"{name} axis '{axis_name}' must be at least 2.")
        normalized.append(axis_value)

    return tuple(normalized)


def _validate_parameter_vector(
    name: str, values: Any, dimensions: int, *, minimum: int
) -> tuple[int, ...]:
    if not isinstance(values, list) or len(values) != dimensions:
        raise ValueError(f"{name} must contain exactly {dimensions} integers.")

    normalized: list[int] = []
    for axis_name, axis_value in zip(AXIS_NAMES[dimensions], values):
        if not isinstance(axis_value, int):
            raise ValueError(f"{name} axis '{axis_name}' must be an integer.")
        if axis_value < minimum:
            raise ValueError(f"{name} axis '{axis_name}' must be at least {minimum}.")
        normalized.append(axis_value)

    return tuple(normalized)


def _validate_bias_values(values: Any, output_channels: int) -> list[float]:
    if not isinstance(values, list) or len(values) != output_channels:
        raise ValueError(
            f"bias_values must contain exactly {output_channels} numbers."
        )

    return [
        _ensure_numeric_scalar(value, "bias_values", (index,))
        for index, value in enumerate(values)
    ]


def _weight_shape(request: ConvolutionRequest | None = None, **kwargs: Any) -> tuple[int, ...]:
    if request is not None:
        operation = request.operation
        input_channels = request.input_channels
        output_channels = request.output_channels
        groups = request.groups
        kernel_shape = request.kernel_shape
    else:
        operation = kwargs["operation"]
        input_channels = kwargs["input_channels"]
        output_channels = kwargs["output_channels"]
        groups = kwargs["groups"]
        kernel_shape = kwargs["kernel_shape"]

    if operation == "conv":
        return (output_channels, input_channels // groups, *kernel_shape)

    return (input_channels, output_channels // groups, *kernel_shape)


def _compute_output_spatial_shape(request: ConvolutionRequest) -> tuple[int, ...]:
    output_shape: list[int] = []

    for axis in range(request.dimensions):
        input_size = request.input_shape[axis]
        kernel_size = request.kernel_shape[axis]
        stride = request.stride[axis]
        padding = request.padding[axis]
        dilation = request.dilation[axis]
        output_padding = request.output_padding[axis]

        if request.operation == "conv":
            numerator = input_size + (2 * padding) - dilation * (kernel_size - 1) - 1
            if numerator < 0:
                raise ValueError(
                    "The selected kernel, padding, and dilation produce an empty output "
                    f"along axis '{AXIS_NAMES[request.dimensions][axis]}'."
                )
            output_size = (numerator // stride) + 1
        else:
            output_size = (
                (input_size - 1) * stride
                - (2 * padding)
                + dilation * (kernel_size - 1)
                + output_padding
                + 1
            )

        if output_size < 1:
            raise ValueError(
                "The selected parameters produce an empty output "
                f"along axis '{AXIS_NAMES[request.dimensions][axis]}'."
            )

        output_shape.append(output_size)

    return tuple(output_shape)


def parse_request(payload: dict[str, Any]) -> ConvolutionRequest:
    dimensions = payload.get("dimensions")
    if dimensions not in (2, 3):
        raise ValueError("Only 2D and 3D visualizations are supported.")

    operation = payload.get("operation", "conv")
    if operation not in ("conv", "conv_transpose"):
        raise ValueError("operation must be 'conv' or 'conv_transpose'.")

    input_shape = _validate_spatial_shape("input_shape", payload.get("input_shape"), dimensions)
    kernel_shape = _validate_spatial_shape(
        "kernel_shape", payload.get("kernel_shape"), dimensions
    )

    input_channels = _validate_positive_int("input_channels", payload.get("input_channels"))
    output_channels = _validate_positive_int(
        "output_channels", payload.get("output_channels")
    )
    groups = _validate_positive_int("groups", payload.get("groups"))
    if input_channels % groups != 0:
        raise ValueError("groups must divide input_channels exactly.")
    if output_channels % groups != 0:
        raise ValueError("groups must divide output_channels exactly.")

    stride = _validate_parameter_vector(
        "stride", payload.get("stride"), dimensions, minimum=1
    )
    padding = _validate_parameter_vector(
        "padding", payload.get("padding"), dimensions, minimum=0
    )
    dilation = _validate_parameter_vector(
        "dilation", payload.get("dilation"), dimensions, minimum=1
    )
    output_padding = _validate_parameter_vector(
        "output_padding", payload.get("output_padding", [0] * dimensions), dimensions, minimum=0
    )

    if operation == "conv" and any(output_padding):
        raise ValueError(
            "output_padding is only used by PyTorch transposed convolution."
        )

    bias_enabled = _validate_bool("bias_enabled", payload.get("bias_enabled"), default=False)
    bias_values = (
        _validate_bias_values(payload.get("bias_values"), output_channels)
        if bias_enabled
        else None
    )

    request = ConvolutionRequest(
        dimensions=dimensions,
        operation=operation,
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        input_channels=input_channels,
        output_channels=output_channels,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        bias_enabled=bias_enabled,
        input_values=[],
        kernel_values=[],
        bias_values=bias_values,
    )

    input_values = _validate_nested_values(
        payload.get("input_values"),
        (input_channels, *input_shape),
        "input_values",
    )
    kernel_values = _validate_nested_values(
        payload.get("kernel_values"),
        _weight_shape(request),
        "kernel_values",
    )

    request = ConvolutionRequest(
        dimensions=request.dimensions,
        operation=request.operation,
        input_shape=request.input_shape,
        kernel_shape=request.kernel_shape,
        input_channels=request.input_channels,
        output_channels=request.output_channels,
        groups=request.groups,
        stride=request.stride,
        padding=request.padding,
        dilation=request.dilation,
        output_padding=request.output_padding,
        bias_enabled=request.bias_enabled,
        input_values=input_values,
        kernel_values=kernel_values,
        bias_values=request.bias_values,
    )

    _compute_output_spatial_shape(request)
    return request


def _to_tensor(values: list[Any]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _build_conv_trace(
    request: ConvolutionRequest,
    input_tensor: torch.Tensor,
    kernel_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    bias_tensor: torch.Tensor | None,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    input_channels_per_group = request.input_channels // request.groups
    output_channels_per_group = request.output_channels // request.groups

    for output_channel in range(request.output_channels):
        group_id = output_channel // output_channels_per_group
        input_channel_start = group_id * input_channels_per_group

        for output_index in _iter_indices(tuple(output_tensor.shape[1:])):
            pre_bias_value = 0.0
            channel_contributions: list[dict[str, Any]] = []
            expression_parts: list[str] = []

            for local_input_channel in range(input_channels_per_group):
                input_channel = input_channel_start + local_input_channel
                sampled_values = torch.zeros(request.kernel_shape, dtype=torch.float64)
                products = torch.zeros(request.kernel_shape, dtype=torch.float64)
                terms: list[dict[str, Any]] = []
                channel_expression_parts: list[str] = []
                kernel_slice = kernel_tensor[(output_channel, local_input_channel)]

                for kernel_index in _iter_indices(request.kernel_shape):
                    input_index = tuple(
                        output_index[axis] * request.stride[axis]
                        - request.padding[axis]
                        + kernel_index[axis] * request.dilation[axis]
                        for axis in range(request.dimensions)
                    )
                    in_bounds = all(
                        0 <= input_index[axis] < request.input_shape[axis]
                        for axis in range(request.dimensions)
                    )
                    if in_bounds:
                        input_value = float(
                            input_tensor[(input_channel, *input_index)].item()
                        )
                        indexed_input = list(input_index)
                    else:
                        input_value = 0.0
                        indexed_input = None

                    kernel_value = float(kernel_slice[kernel_index].item())
                    product_value = input_value * kernel_value

                    sampled_values[kernel_index] = input_value
                    products[kernel_index] = product_value
                    pre_bias_value += product_value
                    channel_expression_parts.append(
                        f"({_format_number(input_value)} x {_format_number(kernel_value)})"
                    )
                    terms.append(
                        {
                            "input_index": indexed_input,
                            "kernel_index": list(kernel_index),
                            "input_value": input_value,
                            "kernel_value": kernel_value,
                            "product": product_value,
                            "padded": not in_bounds,
                        }
                    )

                expression_parts.append(
                    f"channel {input_channel}: " + " + ".join(channel_expression_parts)
                )
                channel_contributions.append(
                    {
                        "input_channel": input_channel,
                        "kernel_channel": local_input_channel,
                        "sampled_values": sampled_values.tolist(),
                        "kernel_values": kernel_slice.tolist(),
                        "products": products.tolist(),
                        "terms": terms,
                        "expression": " + ".join(channel_expression_parts),
                    }
                )

            bias_value = (
                float(bias_tensor[output_channel].item()) if bias_tensor is not None else None
            )
            expression = " + ".join(expression_parts) if expression_parts else "0"
            if bias_value is not None:
                expression = f"{expression} + bias({_format_number(bias_value)})"

            trace.append(
                {
                    "mode": "conv",
                    "group": group_id,
                    "output_channel": output_channel,
                    "output_index": list(output_index),
                    "full_output_index": [output_channel, *output_index],
                    "output_value": float(output_tensor[(output_channel, *output_index)].item()),
                    "pre_bias_value": pre_bias_value,
                    "bias_value": bias_value,
                    "expression": expression,
                    "channel_contributions": channel_contributions,
                }
            )

    return trace


def _build_conv_transpose_trace(
    request: ConvolutionRequest,
    input_tensor: torch.Tensor,
    kernel_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    bias_tensor: torch.Tensor | None,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    input_channels_per_group = request.input_channels // request.groups
    output_channels_per_group = request.output_channels // request.groups

    for output_channel in range(request.output_channels):
        group_id = output_channel // output_channels_per_group
        output_channel_in_group = output_channel - group_id * output_channels_per_group
        input_channel_start = group_id * input_channels_per_group

        for output_index in _iter_indices(tuple(output_tensor.shape[1:])):
            pre_bias_value = 0.0
            channel_contributions: list[dict[str, Any]] = []
            expression_parts: list[str] = []

            for local_input_channel in range(input_channels_per_group):
                input_channel = input_channel_start + local_input_channel
                kernel_slice = kernel_tensor[(input_channel, output_channel_in_group)]
                terms: list[dict[str, Any]] = []
                channel_expression_parts: list[str] = []

                for input_index in _iter_indices(request.input_shape):
                    input_value = float(input_tensor[(input_channel, *input_index)].item())
                    for kernel_index in _iter_indices(request.kernel_shape):
                        projected_output = tuple(
                            input_index[axis] * request.stride[axis]
                            - request.padding[axis]
                            + kernel_index[axis] * request.dilation[axis]
                            for axis in range(request.dimensions)
                        )
                        if projected_output != output_index:
                            continue

                        kernel_value = float(kernel_slice[kernel_index].item())
                        product_value = input_value * kernel_value
                        pre_bias_value += product_value
                        channel_expression_parts.append(
                            f"({_format_number(input_value)} x {_format_number(kernel_value)})"
                        )
                        terms.append(
                            {
                                "input_index": list(input_index),
                                "kernel_index": list(kernel_index),
                                "input_value": input_value,
                                "kernel_value": kernel_value,
                                "product": product_value,
                            }
                        )

                if channel_expression_parts:
                    expression_parts.append(
                        f"channel {input_channel}: " + " + ".join(channel_expression_parts)
                    )

                channel_contributions.append(
                    {
                        "input_channel": input_channel,
                        "kernel_channel": output_channel_in_group,
                        "kernel_values": kernel_slice.tolist(),
                        "terms": terms,
                        "expression": (
                            " + ".join(channel_expression_parts)
                            if channel_expression_parts
                            else "No aligned contributions for this output position."
                        ),
                    }
                )

            bias_value = (
                float(bias_tensor[output_channel].item()) if bias_tensor is not None else None
            )
            expression = " + ".join(expression_parts) if expression_parts else "0"
            if bias_value is not None:
                expression = f"{expression} + bias({_format_number(bias_value)})"

            trace.append(
                {
                    "mode": "conv_transpose",
                    "group": group_id,
                    "output_channel": output_channel,
                    "output_index": list(output_index),
                    "full_output_index": [output_channel, *output_index],
                    "output_value": float(output_tensor[(output_channel, *output_index)].item()),
                    "pre_bias_value": pre_bias_value,
                    "bias_value": bias_value,
                    "expression": expression,
                    "channel_contributions": channel_contributions,
                }
            )

    return trace


def _build_trace(
    request: ConvolutionRequest,
    input_tensor: torch.Tensor,
    kernel_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    bias_tensor: torch.Tensor | None,
) -> list[dict[str, Any]]:
    if request.operation == "conv":
        return _build_conv_trace(
            request, input_tensor, kernel_tensor, output_tensor, bias_tensor
        )
    return _build_conv_transpose_trace(
        request, input_tensor, kernel_tensor, output_tensor, bias_tensor
    )


def solve_convolution(request: ConvolutionRequest) -> dict[str, Any]:
    input_tensor = _to_tensor(request.input_values)
    kernel_tensor = _to_tensor(request.kernel_values)
    bias_tensor = (
        torch.tensor(request.bias_values, dtype=torch.float64)
        if request.bias_enabled and request.bias_values is not None
        else None
    )

    try:
        if request.operation == "conv":
            if request.dimensions == 2:
                output = F.conv2d(
                    input_tensor.unsqueeze(0),
                    kernel_tensor,
                    bias=bias_tensor,
                    stride=request.stride,
                    padding=request.padding,
                    dilation=request.dilation,
                    groups=request.groups,
                )[0]
            else:
                output = F.conv3d(
                    input_tensor.unsqueeze(0),
                    kernel_tensor,
                    bias=bias_tensor,
                    stride=request.stride,
                    padding=request.padding,
                    dilation=request.dilation,
                    groups=request.groups,
                )[0]
        else:
            if request.dimensions == 2:
                output = F.conv_transpose2d(
                    input_tensor.unsqueeze(0),
                    kernel_tensor,
                    bias=bias_tensor,
                    stride=request.stride,
                    padding=request.padding,
                    output_padding=request.output_padding,
                    groups=request.groups,
                    dilation=request.dilation,
                )[0]
            else:
                output = F.conv_transpose3d(
                    input_tensor.unsqueeze(0),
                    kernel_tensor,
                    bias=bias_tensor,
                    stride=request.stride,
                    padding=request.padding,
                    output_padding=request.output_padding,
                    groups=request.groups,
                    dilation=request.dilation,
                )[0]
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc

    output_spatial_shape = tuple(output.shape[1:])
    trace = _build_trace(request, input_tensor, kernel_tensor, output, bias_tensor)

    if request.operation == "conv":
        note = (
            "PyTorch convolution uses cross-correlation semantics, so the kernel "
            "is not flipped."
        )
    else:
        note = (
            "PyTorch transposed convolution expands the spatial layout according "
            "to stride, padding, dilation, and output_padding."
        )

    return {
        "dimensions": request.dimensions,
        "operation": request.operation,
        "operation_label": OPERATION_NAMES[request.operation],
        "axis_names": list(AXIS_NAMES[request.dimensions]),
        "input_shape": list(request.input_shape),
        "kernel_shape": list(request.kernel_shape),
        "input_channels": request.input_channels,
        "output_channels": request.output_channels,
        "groups": request.groups,
        "input_tensor_shape": [request.input_channels, *request.input_shape],
        "weight_shape": list(_weight_shape(request)),
        "output_spatial_shape": list(output_spatial_shape),
        "output_tensor_shape": list(output.shape),
        "input_values": request.input_values,
        "kernel_values": request.kernel_values,
        "bias_enabled": request.bias_enabled,
        "bias_values": request.bias_values,
        "output_values": output.tolist(),
        "trace": trace,
        "semantics": {
            "operation": FUNCTION_NAMES[(request.operation, request.dimensions)],
            "stride": list(request.stride),
            "padding": list(request.padding),
            "dilation": list(request.dilation),
            "output_padding": list(request.output_padding),
            "groups": request.groups,
            "bias_enabled": request.bias_enabled,
            "note": note,
        },
    }
