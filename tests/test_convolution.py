from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from convolution_visualizer.convolution import parse_request, solve_convolution


class ConvolutionEngineTests(unittest.TestCase):
    def test_parameterized_conv2d_matches_pytorch(self) -> None:
        payload = {
            "dimensions": 2,
            "operation": "conv",
            "input_shape": [4, 4],
            "kernel_shape": [2, 2],
            "input_channels": 1,
            "output_channels": 1,
            "groups": 1,
            "stride": [2, 1],
            "padding": [1, 0],
            "dilation": [1, 1],
            "output_padding": [0, 0],
            "bias_enabled": True,
            "input_values": [
                [
                    [1, 2, 3, 0],
                    [0, 1, -1, 4],
                    [2, 3, 1, 2],
                    [1, 0, 2, 1],
                ]
            ],
            "kernel_values": [
                [
                    [
                        [1, -1],
                        [2, 0],
                    ]
                ]
            ],
            "bias_values": [0.5],
        }

        parsed = parse_request(payload)
        result = solve_convolution(parsed)

        expected = F.conv2d(
            torch.tensor(payload["input_values"], dtype=torch.float64).unsqueeze(0),
            torch.tensor(payload["kernel_values"], dtype=torch.float64),
            bias=torch.tensor(payload["bias_values"], dtype=torch.float64),
            stride=tuple(payload["stride"]),
            padding=tuple(payload["padding"]),
            dilation=tuple(payload["dilation"]),
            groups=payload["groups"],
        )[0]

        self.assertEqual(result["output_tensor_shape"], list(expected.shape))
        self.assertEqual(result["output_values"], expected.tolist())
        self.assertEqual(result["semantics"]["stride"], payload["stride"])
        self.assertTrue(result["bias_enabled"])
        self.assertAlmostEqual(result["trace"][0]["bias_value"], 0.5)

    def test_grouped_conv2d_matches_pytorch(self) -> None:
        payload = {
            "dimensions": 2,
            "operation": "conv",
            "input_shape": [3, 3],
            "kernel_shape": [2, 2],
            "input_channels": 2,
            "output_channels": 2,
            "groups": 2,
            "stride": [1, 1],
            "padding": [0, 0],
            "dilation": [1, 1],
            "output_padding": [0, 0],
            "bias_enabled": False,
            "input_values": [
                [
                    [1, 2, 0],
                    [3, 1, 2],
                    [0, 1, 3],
                ],
                [
                    [2, 1, 0],
                    [1, 0, 1],
                    [2, 3, 1],
                ],
            ],
            "kernel_values": [
                [
                    [
                        [1, 0],
                        [-1, 2],
                    ]
                ],
                [
                    [
                        [0, 1],
                        [2, -1],
                    ]
                ],
            ],
            "bias_values": [0, 0],
        }

        parsed = parse_request(payload)
        result = solve_convolution(parsed)

        expected = F.conv2d(
            torch.tensor(payload["input_values"], dtype=torch.float64).unsqueeze(0),
            torch.tensor(payload["kernel_values"], dtype=torch.float64),
            bias=None,
            stride=tuple(payload["stride"]),
            padding=tuple(payload["padding"]),
            dilation=tuple(payload["dilation"]),
            groups=payload["groups"],
        )[0]

        self.assertEqual(result["output_values"], expected.tolist())
        self.assertEqual(result["weight_shape"], [2, 1, 2, 2])
        self.assertEqual(result["trace"][0]["channel_contributions"][0]["input_channel"], 0)
        second_output_channel_trace = next(
            entry for entry in result["trace"] if entry["output_channel"] == 1
        )
        self.assertEqual(
            second_output_channel_trace["channel_contributions"][0]["input_channel"], 1
        )

    def test_conv_transpose2d_matches_pytorch(self) -> None:
        payload = {
            "dimensions": 2,
            "operation": "conv_transpose",
            "input_shape": [3, 3],
            "kernel_shape": [2, 2],
            "input_channels": 1,
            "output_channels": 1,
            "groups": 1,
            "stride": [2, 2],
            "padding": [0, 0],
            "dilation": [1, 1],
            "output_padding": [1, 1],
            "bias_enabled": True,
            "input_values": [
                [
                    [1, 2, 1],
                    [0, 1, 2],
                    [1, 0, 1],
                ]
            ],
            "kernel_values": [
                [
                    [
                        [1, 0],
                        [0, 1],
                    ]
                ]
            ],
            "bias_values": [1.0],
        }

        parsed = parse_request(payload)
        result = solve_convolution(parsed)

        expected = F.conv_transpose2d(
            torch.tensor(payload["input_values"], dtype=torch.float64).unsqueeze(0),
            torch.tensor(payload["kernel_values"], dtype=torch.float64),
            bias=torch.tensor(payload["bias_values"], dtype=torch.float64),
            stride=tuple(payload["stride"]),
            padding=tuple(payload["padding"]),
            output_padding=tuple(payload["output_padding"]),
            groups=payload["groups"],
            dilation=tuple(payload["dilation"]),
        )[0]

        self.assertEqual(result["output_tensor_shape"], list(expected.shape))
        self.assertEqual(result["output_values"], expected.tolist())
        self.assertEqual(result["trace"][0]["mode"], "conv_transpose")

    def test_conv3d_matches_pytorch(self) -> None:
        payload = {
            "dimensions": 3,
            "operation": "conv",
            "input_shape": [3, 3, 3],
            "kernel_shape": [2, 2, 2],
            "input_channels": 1,
            "output_channels": 1,
            "groups": 1,
            "stride": [1, 1, 1],
            "padding": [0, 0, 0],
            "dilation": [1, 1, 1],
            "output_padding": [0, 0, 0],
            "bias_enabled": False,
            "input_values": [
                [
                    [
                        [1, 0, 2],
                        [2, 1, 0],
                        [1, 3, 1],
                    ],
                    [
                        [0, 2, 1],
                        [3, 1, 2],
                        [2, 0, 1],
                    ],
                    [
                        [1, 1, 0],
                        [0, 2, 3],
                        [2, 1, 1],
                    ],
                ]
            ],
            "kernel_values": [
                [
                    [
                        [
                            [1, -1],
                            [0, 2],
                        ],
                        [
                            [2, 0],
                            [-1, 1],
                        ],
                    ],
                ]
            ],
            "bias_values": [0],
        }

        parsed = parse_request(payload)
        result = solve_convolution(parsed)

        expected = F.conv3d(
            torch.tensor(payload["input_values"], dtype=torch.float64).unsqueeze(0),
            torch.tensor(payload["kernel_values"], dtype=torch.float64),
            bias=None,
            stride=tuple(payload["stride"]),
            padding=tuple(payload["padding"]),
            dilation=tuple(payload["dilation"]),
            groups=payload["groups"],
        )[0]

        self.assertEqual(result["output_values"], expected.tolist())
        self.assertEqual(len(result["trace"]), 8)

    def test_rejects_output_padding_for_standard_conv(self) -> None:
        payload = {
            "dimensions": 2,
            "operation": "conv",
            "input_shape": [3, 3],
            "kernel_shape": [2, 2],
            "input_channels": 1,
            "output_channels": 1,
            "groups": 1,
            "stride": [1, 1],
            "padding": [0, 0],
            "dilation": [1, 1],
            "output_padding": [1, 0],
            "bias_enabled": False,
            "input_values": [[[1, 2, 3], [0, 1, 0], [2, 1, 2]]],
            "kernel_values": [[[[1, 0], [0, 1]]]],
            "bias_values": [0],
        }

        with self.assertRaises(ValueError):
            parse_request(payload)


if __name__ == "__main__":
    unittest.main()
