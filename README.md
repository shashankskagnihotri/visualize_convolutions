# Convolution Studio

An interactive browser-based learning tool for bachelor students learning convolution for the first time.

The app launches from Python, opens in your browser, and lets you build tensors, weights, and bias values by hand. It computes the real result with PyTorch and then explains the operation with readable arrays, clickable output values, and an animation that walks through the output tensor step by step.

## What It Does

- Uses real PyTorch ops for the final answer:
  - `torch.nn.functional.conv2d`
  - `torch.nn.functional.conv3d`
  - `torch.nn.functional.conv_transpose2d`
  - `torch.nn.functional.conv_transpose3d`
- Supports 2D by default, with optional 3D mode.
- Supports custom spatial sizes starting at `2 x 2` or `2 x 2 x 2`.
- Supports custom input channels, output channels, grouped convolution, and optional bias.
- Lets learners control stride, padding, dilation, and groups.
- Exposes `output_padding` correctly through transposed convolution mode, which is where PyTorch uses it.
- Shows the output tensor in two ways:
  - As a clean tensor/array view
  - As a clickable output explorer
- Provides a detailed trace for every output value.
- Includes a step-through animation so students can watch the operator fill the output tensor.

## PyTorch Semantics

- Standard convolution in PyTorch uses cross-correlation semantics, so the kernel is not flipped.
- `output_padding` is not a parameter of standard convolution. It is a transposed-convolution setting, so the UI enables it only in transposed mode.
- Batch size is fixed to `1` in this teaching app so the focus stays on channels and spatial dimensions.

## Setup

You already created the environment:

```bash
conda create -n learn_convolution python=3.12 -y
```

Activate it:

```bash
conda activate learn_convolution
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Run

Start the app on the default port:

```bash
python -W ignore main.py
```

Or choose a custom port:

```bash
python -W ignore main.py --port 6009
python -W ignore main.py --port 6010
```

The app will try to open automatically in your browser. If it does not, visit:

```text
http://127.0.0.1:6009
```

## Testing

Run the verification suite with:

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- Parameterized `conv2d`
- Grouped convolution
- `conv_transpose2d` with `output_padding`
- `conv3d`
- Validation of incorrect parameter combinations

## Project Structure

```text
main.py
convolution_visualizer/
  app.py
  convolution.py
  templates/index.html
  static/styles.css
  static/app.js
tests/
  test_convolution.py
```
