const axisLabels = {
  2: ["row", "column"],
  3: ["depth", "row", "column"],
};

const operationLabels = {
  conv: "Convolution",
  conv_transpose: "Transposed convolution",
};

const examplePresets = {
  "conv-2": {
    dimensions: 2,
    operation: "conv",
    inputShape: [5, 5],
    kernelShape: [3, 3],
    inputChannels: 1,
    outputChannels: 1,
    groups: 1,
    stride: [1, 1],
    padding: [1, 1],
    dilation: [1, 1],
    outputPadding: [0, 0],
    biasEnabled: false,
    inputValues: [
      [
        [0, 1, 2, 1, 0],
        [2, 3, 1, 0, 1],
        [1, 2, 4, 2, 1],
        [0, 1, 2, 3, 2],
        [1, 0, 1, 2, 1],
      ],
    ],
    kernelValues: [
      [
        [
          [1, 0, -1],
          [1, 0, -1],
          [1, 0, -1],
        ],
      ],
    ],
    biasValues: [0],
  },
  "conv-3": {
    dimensions: 3,
    operation: "conv",
    inputShape: [3, 3, 3],
    kernelShape: [2, 2, 2],
    inputChannels: 1,
    outputChannels: 1,
    groups: 1,
    stride: [1, 1, 1],
    padding: [0, 0, 0],
    dilation: [1, 1, 1],
    outputPadding: [0, 0, 0],
    biasEnabled: true,
    inputValues: [
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
      ],
    ],
    kernelValues: [
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
      ],
    ],
    biasValues: [1],
  },
  "conv_transpose-2": {
    dimensions: 2,
    operation: "conv_transpose",
    inputShape: [3, 3],
    kernelShape: [2, 2],
    inputChannels: 1,
    outputChannels: 1,
    groups: 1,
    stride: [2, 2],
    padding: [0, 0],
    dilation: [1, 1],
    outputPadding: [1, 1],
    biasEnabled: true,
    inputValues: [
      [
        [1, 2, 1],
        [0, 1, 2],
        [1, 0, 1],
      ],
    ],
    kernelValues: [
      [
        [
          [1, 0],
          [0, 1],
        ],
      ],
    ],
    biasValues: [0],
  },
  "conv_transpose-3": {
    dimensions: 3,
    operation: "conv_transpose",
    inputShape: [2, 2, 2],
    kernelShape: [2, 2, 2],
    inputChannels: 1,
    outputChannels: 1,
    groups: 1,
    stride: [2, 2, 2],
    padding: [0, 0, 0],
    dilation: [1, 1, 1],
    outputPadding: [1, 1, 1],
    biasEnabled: false,
    inputValues: [
      [
        [
          [1, 0],
          [2, 1],
        ],
        [
          [0, 1],
          [1, 2],
        ],
      ],
    ],
    kernelValues: [
      [
        [
          [
            [1, 0],
            [0, 1],
          ],
          [
            [1, -1],
            [0, 1],
          ],
        ],
      ],
    ],
    biasValues: [0],
  },
};

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function formatNumber(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return "NaN";
  }
  const rounded = Math.round(numeric * 1e6) / 1e6;
  if (Number.isInteger(rounded)) {
    return String(rounded);
  }
  return rounded.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function createFilledTensor(shape, valueFactory, path = []) {
  if (shape.length === 0) {
    return valueFactory(path);
  }
  return Array.from({ length: shape[0] }, (_, index) =>
    createFilledTensor(shape.slice(1), valueFactory, [...path, index]),
  );
}

function iterateIndices(shape) {
  const output = [];

  function visit(level, prefix) {
    if (level === shape.length) {
      output.push(prefix);
      return;
    }
    for (let index = 0; index < shape[level]; index += 1) {
      visit(level + 1, [...prefix, index]);
    }
  }

  visit(0, []);
  return output;
}

function getValueAt(values, indices) {
  return indices.reduce((accumulator, index) => accumulator[index], values);
}

function setValueAt(values, indices, nextValue) {
  let cursor = values;
  for (let index = 0; index < indices.length - 1; index += 1) {
    cursor = cursor[indices[index]];
  }
  cursor[indices[indices.length - 1]] = nextValue;
}

function resizeTensor(oldTensor, oldShape, newShape, fillValue = 0) {
  return createFilledTensor(newShape, (path) => {
    const withinBounds = path.every((index, axis) => index < (oldShape[axis] ?? 0));
    return withinBounds ? getValueAt(oldTensor, path) : fillValue;
  });
}

function resizeVector(oldVector, newLength, fillValue = 0) {
  return Array.from({ length: newLength }, (_, index) =>
    index < oldVector.length ? oldVector[index] : fillValue,
  );
}

function describeShape(shape) {
  return shape.join(" x ");
}

function toIndexKey(indices) {
  return indices.join(",");
}

function buildHighlightSet(indicesList) {
  const set = new Set();
  indicesList.forEach((indices) => {
    if (indices) {
      set.add(toIndexKey(indices));
    }
  });
  return set;
}

function currentPresetKey(dimensions = null, operation = null) {
  return `${operation || state.operation}-${dimensions || state.dimensions}`;
}

function getPreset(dimensions = null, operation = null) {
  return examplePresets[currentPresetKey(dimensions, operation)];
}

function getValidGroups(inputChannels, outputChannels) {
  const options = [];
  const maxGroup = Math.min(inputChannels, outputChannels);
  for (let value = 1; value <= maxGroup; value += 1) {
    if (inputChannels % value === 0 && outputChannels % value === 0) {
      options.push(value);
    }
  }
  return options;
}

function getInputTensorShape(stateLike = state) {
  return [stateLike.inputChannels, ...stateLike.inputShape];
}

function getWeightShape(stateLike = state) {
  if (stateLike.operation === "conv") {
    return [
      stateLike.outputChannels,
      stateLike.inputChannels / stateLike.groups,
      ...stateLike.kernelShape,
    ];
  }

  return [
    stateLike.inputChannels,
    stateLike.outputChannels / stateLike.groups,
    ...stateLike.kernelShape,
  ];
}

function previewOutputSpatialShape(stateLike = state) {
  const output = [];

  for (let axis = 0; axis < stateLike.dimensions; axis += 1) {
    const inputSize = stateLike.inputShape[axis];
    const kernelSize = stateLike.kernelShape[axis];
    const stride = stateLike.stride[axis];
    const padding = stateLike.padding[axis];
    const dilation = stateLike.dilation[axis];
    const outputPadding = stateLike.outputPadding[axis];

    let size = null;
    if (stateLike.operation === "conv") {
      const numerator = inputSize + 2 * padding - dilation * (kernelSize - 1) - 1;
      if (numerator < 0) {
        return null;
      }
      size = Math.floor(numerator / stride) + 1;
    } else {
      size =
        (inputSize - 1) * stride -
        2 * padding +
        dilation * (kernelSize - 1) +
        outputPadding +
        1;
    }

    if (size < 1) {
      return null;
    }
    output.push(size);
  }

  return output;
}

function previewOutputTensorShape(stateLike = state) {
  const spatial = previewOutputSpatialShape(stateLike);
  if (!spatial) {
    return null;
  }
  return [stateLike.outputChannels, ...spatial];
}

function loadPreset(dimensions = state.dimensions, operation = state.operation) {
  const preset = getPreset(dimensions, operation);
  state.dimensions = preset.dimensions;
  state.operation = preset.operation;
  state.inputShape = [...preset.inputShape];
  state.kernelShape = [...preset.kernelShape];
  state.inputChannels = preset.inputChannels;
  state.outputChannels = preset.outputChannels;
  state.groups = preset.groups;
  state.stride = [...preset.stride];
  state.padding = [...preset.padding];
  state.dilation = [...preset.dilation];
  state.outputPadding = [...preset.outputPadding];
  state.biasEnabled = preset.biasEnabled;
  state.inputValues = deepClone(preset.inputValues);
  state.kernelValues = deepClone(preset.kernelValues);
  state.biasValues = deepClone(preset.biasValues);
}

function snapshotStateShapes() {
  return {
    inputTensorShape: getInputTensorShape(),
    weightShape: getWeightShape(),
    outputChannels: state.outputChannels,
  };
}

function normalizeStateFromSnapshot(snapshot, options = {}) {
  const resetKernel = options.resetKernel || false;
  const validGroups = getValidGroups(state.inputChannels, state.outputChannels);
  if (!validGroups.includes(state.groups)) {
    state.groups = validGroups[0];
  }

  state.inputValues = resizeTensor(
    state.inputValues,
    snapshot.inputTensorShape,
    getInputTensorShape(),
    0,
  );
  state.kernelValues = resetKernel
    ? createFilledTensor(getWeightShape(), () => 0)
    : resizeTensor(state.kernelValues, snapshot.weightShape, getWeightShape(), 0);
  state.biasValues = resizeVector(state.biasValues, state.outputChannels, 0);

  if (state.operation === "conv") {
    state.outputPadding = Array.from({ length: state.dimensions }, () => 0);
  }
}

const state = {
  dimensions: 2,
  operation: "conv",
  inputShape: [],
  kernelShape: [],
  inputChannels: 1,
  outputChannels: 1,
  groups: 1,
  stride: [],
  padding: [],
  dilation: [],
  outputPadding: [],
  biasEnabled: false,
  inputValues: [],
  kernelValues: [],
  biasValues: [],
  latestResult: null,
  pendingTimer: null,
  animationFrames: [],
  animationIndex: 0,
  animationTimer: null,
  animationPlaying: false,
};

loadPreset(2, "conv");

const elements = {
  dimensions: document.getElementById("dimensions"),
  operation: document.getElementById("operation"),
  operationHelp: document.getElementById("operation-help"),
  inputShapeControls: document.getElementById("input-shape-controls"),
  kernelShapeControls: document.getElementById("kernel-shape-controls"),
  strideControls: document.getElementById("stride-controls"),
  paddingControls: document.getElementById("padding-controls"),
  dilationControls: document.getElementById("dilation-controls"),
  outputPaddingControls: document.getElementById("output-padding-controls"),
  outputPaddingBlock: document.getElementById("output-padding-block"),
  inputChannels: document.getElementById("input-channels"),
  outputChannels: document.getElementById("output-channels"),
  groups: document.getElementById("groups"),
  biasEnabled: document.getElementById("bias-enabled"),
  inputEditor: document.getElementById("input-editor"),
  kernelEditor: document.getElementById("kernel-editor"),
  biasEditor: document.getElementById("bias-editor"),
  biasCard: document.getElementById("bias-card"),
  inputShapeLabel: document.getElementById("input-shape-label"),
  kernelShapeLabel: document.getElementById("kernel-shape-label"),
  biasShapeLabel: document.getElementById("bias-shape-label"),
  outputShapeBadge: document.getElementById("output-shape-badge"),
  problemSummary: document.getElementById("problem-summary"),
  resultError: document.getElementById("result-error"),
  resultSummary: document.getElementById("result-summary"),
  outputArrayRender: document.getElementById("output-array-render"),
  outputCardRender: document.getElementById("output-card-render"),
  semanticsPill: document.getElementById("semantics-pill"),
  calculateButton: document.getElementById("calculate-button"),
  generateAnimationButton: document.getElementById("generate-animation-button"),
  animationPanel: document.getElementById("animation-panel"),
  prevFrameButton: document.getElementById("prev-frame-button"),
  nextFrameButton: document.getElementById("next-frame-button"),
  playAnimationButton: document.getElementById("play-animation-button"),
  animationStatus: document.getElementById("animation-status"),
  animationCaption: document.getElementById("animation-caption"),
  animationInputRender: document.getElementById("animation-input-render"),
  animationKernelRender: document.getElementById("animation-kernel-render"),
  animationOutputRender: document.getElementById("animation-output-render"),
  traceRender: document.getElementById("trace-render"),
  exampleButton: document.getElementById("example-button"),
  randomizeButton: document.getElementById("randomize-button"),
};

function summaryPill(label, value) {
  const pill = document.createElement("span");
  pill.className = "summary-pill";
  pill.textContent = `${label}: ${value}`;
  return pill;
}

function stopAnimationPlayback() {
  if (state.animationTimer) {
    window.clearInterval(state.animationTimer);
    state.animationTimer = null;
  }
  state.animationPlaying = false;
  elements.playAnimationButton.textContent = "Play";
}

function clearAnimation() {
  stopAnimationPlayback();
  state.animationFrames = [];
  state.animationIndex = 0;
  elements.animationPanel.classList.add("hidden");
  elements.animationInputRender.replaceChildren();
  elements.animationKernelRender.replaceChildren();
  elements.animationOutputRender.replaceChildren();
  elements.animationCaption.textContent = "";
  elements.animationStatus.textContent = "Frame --";
}

function invalidateComputedViews() {
  state.latestResult = null;
  elements.generateAnimationButton.disabled = true;
  elements.resultError.classList.add("hidden");
  elements.resultSummary.textContent = "";
  elements.semanticsPill.textContent = "PyTorch semantics";
  elements.outputArrayRender.replaceChildren();
  elements.outputCardRender.replaceChildren();
  elements.traceRender.replaceChildren();
  clearAnimation();
}

function randomValue() {
  const values = [-2, -1, 0, 1, 2, 3];
  return values[Math.floor(Math.random() * values.length)];
}

function renderAxisInputs(target, values, minimum, onChange) {
  target.replaceChildren();
  axisLabels[state.dimensions].forEach((axisName, axis) => {
    const wrapper = document.createElement("div");
    wrapper.className = "shape-field";

    const label = document.createElement("span");
    label.textContent = axisName;

    const input = document.createElement("input");
    input.type = "number";
    input.min = String(minimum);
    input.step = "1";
    input.value = String(values[axis]);
    input.addEventListener("input", () => {
      const parsed = Number.parseInt(input.value || String(minimum), 10);
      onChange(axis, Math.max(minimum, Number.isFinite(parsed) ? parsed : minimum));
    });

    wrapper.append(label, input);
    target.append(wrapper);
  });
}

function renderGroupsSelect() {
  const options = getValidGroups(state.inputChannels, state.outputChannels);
  if (!options.includes(state.groups)) {
    state.groups = options[0];
  }

  elements.groups.replaceChildren();
  options.forEach((groupValue) => {
    const option = document.createElement("option");
    option.value = String(groupValue);
    option.textContent = String(groupValue);
    if (groupValue === state.groups) {
      option.selected = true;
    }
    elements.groups.append(option);
  });
}

function buildMatrixEditor(rootValues, matrixShape, pathPrefix, labelPrefix) {
  const table = document.createElement("table");
  table.className = "matrix-table";

  for (let row = 0; row < matrixShape[0]; row += 1) {
    const tr = document.createElement("tr");
    for (let column = 0; column < matrixShape[1]; column += 1) {
      const td = document.createElement("td");
      const input = document.createElement("input");
      const fullPath = [...pathPrefix, row, column];
      input.className = "matrix-input";
      input.type = "number";
      input.step = "any";
      input.value = formatNumber(getValueAt(rootValues, fullPath));
      input.setAttribute("aria-label", `${labelPrefix} ${fullPath.join(",")}`);
      input.addEventListener("input", () => {
        const numeric = Number.parseFloat(input.value);
        setValueAt(rootValues, fullPath, Number.isFinite(numeric) ? numeric : 0);
        queueAutoCompute();
      });
      td.append(input);
      tr.append(td);
    }
    table.append(tr);
  }

  return table;
}

function buildMatrixPreview(matrixValues, highlightSet, targetSet, pathPrefix) {
  const table = document.createElement("table");
  table.className = "matrix-table";

  for (let row = 0; row < matrixValues.length; row += 1) {
    const tr = document.createElement("tr");
    for (let column = 0; column < matrixValues[row].length; column += 1) {
      const td = document.createElement("td");
      const span = document.createElement("span");
      const absoluteIndex = [...pathPrefix, row, column];
      const key = toIndexKey(absoluteIndex);
      span.className = "matrix-value";
      if (highlightSet && highlightSet.has(key)) {
        span.classList.add("is-highlight");
      }
      if (targetSet && targetSet.has(key)) {
        span.classList.add("is-target");
      }
      span.textContent = formatNumber(matrixValues[row][column]);
      td.append(span);
      tr.append(td);
    }
    table.append(tr);
  }

  return table;
}

function appendSpatialEditor(container, rootValues, spatialShape, pathPrefix, labelPrefix, title) {
  if (state.dimensions === 2) {
    const block = document.createElement("div");
    block.className = "mini-grid-card";
    const heading = document.createElement("h4");
    heading.textContent = title;
    block.append(heading, buildMatrixEditor(rootValues, spatialShape, pathPrefix, labelPrefix));
    container.append(block);
    return;
  }

  const stack = document.createElement("div");
  stack.className = "depth-stack";

  for (let depth = 0; depth < spatialShape[0]; depth += 1) {
    const block = document.createElement("div");
    block.className = "mini-grid-card";
    const heading = document.createElement("h4");
    heading.textContent = `${title} | depth ${depth}`;
    block.append(
      heading,
      buildMatrixEditor(
        rootValues,
        spatialShape.slice(1),
        [...pathPrefix, depth],
        `${labelPrefix} depth ${depth}`,
      ),
    );
    stack.append(block);
  }

  container.append(stack);
}

function appendSpatialPreview(
  container,
  spatialValues,
  spatialShape,
  title,
  highlightSet = null,
  targetSet = null,
) {
  if (spatialShape.length === 2) {
    const block = document.createElement("div");
    block.className = "mini-grid-card";
    const heading = document.createElement("h4");
    heading.textContent = title;
    block.append(heading, buildMatrixPreview(spatialValues, highlightSet, targetSet, []));
    container.append(block);
    return;
  }

  const stack = document.createElement("div");
  stack.className = "depth-stack";

  for (let depth = 0; depth < spatialShape[0]; depth += 1) {
    const block = document.createElement("div");
    block.className = "mini-grid-card";
    const heading = document.createElement("h4");
    heading.textContent = `${title} | depth ${depth}`;
    block.append(
      heading,
      buildMatrixPreview(
        spatialValues[depth],
        highlightSet,
        targetSet,
        [depth],
      ),
    );
    stack.append(block);
  }

  container.append(stack);
}

function renderInputEditor() {
  elements.inputEditor.replaceChildren();
  for (let inputChannel = 0; inputChannel < state.inputChannels; inputChannel += 1) {
    const card = document.createElement("div");
    card.className = "slice-card";
    const heading = document.createElement("h4");
    heading.textContent = `Input channel ${inputChannel}`;
    card.append(heading);
    appendSpatialEditor(
      card,
      state.inputValues,
      state.inputShape,
      [inputChannel],
      `Input channel ${inputChannel}`,
      "Array values",
    );
    elements.inputEditor.append(card);
  }
}

function kernelPathFor(outputChannel, inputChannel) {
  const outputChannelsPerGroup = state.outputChannels / state.groups;
  const inputChannelsPerGroup = state.inputChannels / state.groups;
  const groupId = Math.floor(outputChannel / outputChannelsPerGroup);
  const inputStart = groupId * inputChannelsPerGroup;
  const outputChannelInGroup = outputChannel - groupId * outputChannelsPerGroup;

  if (state.operation === "conv") {
    return [outputChannel, inputChannel - inputStart];
  }
  return [inputChannel, outputChannelInGroup];
}

function renderKernelEditor() {
  elements.kernelEditor.replaceChildren();
  const outputChannelsPerGroup = state.outputChannels / state.groups;
  const inputChannelsPerGroup = state.inputChannels / state.groups;

  for (let outputChannel = 0; outputChannel < state.outputChannels; outputChannel += 1) {
    const card = document.createElement("div");
    card.className = "slice-card";
    const heading = document.createElement("h4");
    const groupId = Math.floor(outputChannel / outputChannelsPerGroup);
    heading.textContent = `Output channel ${outputChannel} | group ${groupId}`;
    card.append(heading);

    const inputStart = groupId * inputChannelsPerGroup;
    for (let offset = 0; offset < inputChannelsPerGroup; offset += 1) {
      const inputChannel = inputStart + offset;
      appendSpatialEditor(
        card,
        state.kernelValues,
        state.kernelShape,
        kernelPathFor(outputChannel, inputChannel),
        `Weight output ${outputChannel} input ${inputChannel}`,
        `Weight slice from input channel ${inputChannel}`,
      );
    }

    elements.kernelEditor.append(card);
  }
}

function renderBiasEditor() {
  elements.biasCard.classList.toggle("hidden", !state.biasEnabled);
  if (!state.biasEnabled) {
    return;
  }

  elements.biasEditor.replaceChildren();
  const vector = document.createElement("div");
  vector.className = "vector-editor";

  for (let outputChannel = 0; outputChannel < state.outputChannels; outputChannel += 1) {
    const item = document.createElement("label");
    item.className = "vector-item";
    const caption = document.createElement("span");
    caption.textContent = `bias[${outputChannel}]`;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = formatNumber(state.biasValues[outputChannel] ?? 0);
    input.addEventListener("input", () => {
      const numeric = Number.parseFloat(input.value);
      state.biasValues[outputChannel] = Number.isFinite(numeric) ? numeric : 0;
      queueAutoCompute();
    });
    item.append(caption, input);
    vector.append(item);
  }

  elements.biasEditor.append(vector);
}

function renderSummary() {
  const weightShape = getWeightShape();
  const outputTensorShape = previewOutputTensorShape();
  elements.problemSummary.replaceChildren(
    summaryPill("Operation", operationLabels[state.operation]),
    summaryPill("Input tensor", describeShape(getInputTensorShape())),
    summaryPill("Weight tensor", describeShape(weightShape)),
    summaryPill("Groups", String(state.groups)),
    summaryPill("Bias", state.biasEnabled ? "enabled" : "disabled"),
    summaryPill(
      "Output preview",
      outputTensorShape ? describeShape(outputTensorShape) : "invalid",
    ),
  );
}

function renderState() {
  elements.dimensions.value = String(state.dimensions);
  elements.operation.value = state.operation;
  elements.inputChannels.value = String(state.inputChannels);
  elements.outputChannels.value = String(state.outputChannels);
  elements.biasEnabled.checked = state.biasEnabled;
  renderGroupsSelect();

  renderAxisInputs(elements.inputShapeControls, state.inputShape, 2, (axis, nextValue) => {
    const snapshot = snapshotStateShapes();
    state.inputShape[axis] = nextValue;
    normalizeStateFromSnapshot(snapshot);
    renderState();
    queueAutoCompute();
  });

  renderAxisInputs(elements.kernelShapeControls, state.kernelShape, 2, (axis, nextValue) => {
    const snapshot = snapshotStateShapes();
    state.kernelShape[axis] = nextValue;
    normalizeStateFromSnapshot(snapshot);
    renderState();
    queueAutoCompute();
  });

  renderAxisInputs(elements.strideControls, state.stride, 1, (axis, nextValue) => {
    state.stride[axis] = nextValue;
    queueAutoCompute();
  });

  renderAxisInputs(elements.paddingControls, state.padding, 0, (axis, nextValue) => {
    state.padding[axis] = nextValue;
    queueAutoCompute();
  });

  renderAxisInputs(elements.dilationControls, state.dilation, 1, (axis, nextValue) => {
    state.dilation[axis] = nextValue;
    queueAutoCompute();
  });

  renderAxisInputs(
    elements.outputPaddingControls,
    state.outputPadding,
    0,
    (axis, nextValue) => {
      state.outputPadding[axis] = nextValue;
      queueAutoCompute();
    },
  );

  elements.outputPaddingBlock.classList.toggle(
    "hidden",
    state.operation !== "conv_transpose",
  );
  elements.operationHelp.textContent =
    state.operation === "conv"
      ? "Standard convolution uses PyTorch cross-correlation semantics."
      : "Transposed convolution grows the spatial layout and uses output padding.";

  elements.inputShapeLabel.textContent = describeShape(getInputTensorShape());
  elements.kernelShapeLabel.textContent = describeShape(getWeightShape());
  elements.biasShapeLabel.textContent = state.biasEnabled
    ? describeShape([state.outputChannels])
    : "disabled";

  const outputPreview = previewOutputTensorShape();
  elements.outputShapeBadge.textContent = outputPreview
    ? `Output shape: ${describeShape(outputPreview)}`
    : "Output shape: invalid";

  renderSummary();
  renderInputEditor();
  renderKernelEditor();
  renderBiasEditor();
}

function buildOutputButtonGrid(channelValues, spatialShape, outputChannel, prefix = []) {
  const wrapper = document.createElement("div");
  wrapper.className = "output-grid";

  iterateIndices(spatialShape).forEach((index) => {
    const button = document.createElement("button");
    const absoluteSpatialIndex = [...prefix, ...index];
    const fullOutputIndex = [outputChannel, ...absoluteSpatialIndex];
    button.type = "button";
    button.className = "output-cell-button";
    button.innerHTML = `<small>${fullOutputIndex.join(", ")}</small>${formatNumber(
      getValueAt(channelValues, absoluteSpatialIndex),
    )}`;
    button.addEventListener("click", () => {
      const traceNode = document.getElementById(`trace-${fullOutputIndex.join("-")}`);
      if (traceNode) {
        traceNode.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
    wrapper.append(button);
  });

  return wrapper;
}

function renderOutputArrays(result) {
  elements.outputArrayRender.replaceChildren();

  for (let outputChannel = 0; outputChannel < result.output_channels; outputChannel += 1) {
    const card = document.createElement("div");
    card.className = "output-slice";
    const heading = document.createElement("h3");
    heading.textContent = `Output channel ${outputChannel}`;
    card.append(heading);
    appendSpatialPreview(
      card,
      result.output_values[outputChannel],
      result.output_spatial_shape,
      "Array view",
    );
    elements.outputArrayRender.append(card);
  }
}

function renderOutputExplorer(result) {
  elements.outputCardRender.replaceChildren();

  for (let outputChannel = 0; outputChannel < result.output_channels; outputChannel += 1) {
    const card = document.createElement("div");
    card.className = "output-slice";
    const heading = document.createElement("h3");
    heading.textContent = `Output channel ${outputChannel}`;
    card.append(heading);

    if (result.dimensions === 2) {
      card.append(
        buildOutputButtonGrid(
          result.output_values[outputChannel],
          result.output_spatial_shape,
          outputChannel,
        ),
      );
    } else {
      for (let depth = 0; depth < result.output_spatial_shape[0]; depth += 1) {
        const subCard = document.createElement("div");
        subCard.className = "mini-grid-card";
        const subHeading = document.createElement("h4");
        subHeading.textContent = `Depth ${depth}`;
        subCard.append(
          subHeading,
          buildOutputButtonGrid(
            result.output_values[outputChannel],
            result.output_spatial_shape.slice(1),
            outputChannel,
            [depth],
          ),
        );
        card.append(subCard);
      }
    }

    elements.outputCardRender.append(card);
  }
}

function traceTitle(entry) {
  return `Output[${entry.full_output_index.join(", ")}]`;
}

function renderTraceContribution(result, entry, contribution) {
  const block = document.createElement("div");
  block.className = "contribution-card";

  const heading = document.createElement("h4");
  heading.textContent = `Input channel ${contribution.input_channel}`;
  block.append(heading);

  const grids = document.createElement("div");
  grids.className = "trace-grids";

  if (entry.mode === "conv") {
    const sourceWrap = document.createElement("div");
    sourceWrap.className = "mini-grids";
    appendSpatialPreview(
      sourceWrap,
      result.input_values[contribution.input_channel],
      result.input_shape,
      "Source channel",
      buildHighlightSet(
        contribution.terms
          .filter((term) => term.input_index !== null)
          .map((term) => term.input_index),
      ),
    );

    const sampledWrap = document.createElement("div");
    sampledWrap.className = "mini-grids";
    appendSpatialPreview(
      sampledWrap,
      contribution.sampled_values,
      result.kernel_shape,
      "Sampled values",
    );

    const kernelWrap = document.createElement("div");
    kernelWrap.className = "mini-grids";
    appendSpatialPreview(
      kernelWrap,
      contribution.kernel_values,
      result.kernel_shape,
      "Kernel slice",
    );

    const productWrap = document.createElement("div");
    productWrap.className = "mini-grids";
    appendSpatialPreview(
      productWrap,
      contribution.products,
      result.kernel_shape,
      "Products",
    );

    grids.append(sourceWrap, sampledWrap, kernelWrap, productWrap);
  } else {
    const sourceWrap = document.createElement("div");
    sourceWrap.className = "mini-grids";
    appendSpatialPreview(
      sourceWrap,
      result.input_values[contribution.input_channel],
      result.input_shape,
      "Contributing input positions",
      buildHighlightSet(contribution.terms.map((term) => term.input_index)),
    );

    const kernelWrap = document.createElement("div");
    kernelWrap.className = "mini-grids";
    appendSpatialPreview(
      kernelWrap,
      contribution.kernel_values,
      result.kernel_shape,
      "Kernel slice",
      buildHighlightSet(contribution.terms.map((term) => term.kernel_index)),
    );

    grids.append(sourceWrap, kernelWrap);
  }

  block.append(grids);

  const formula = document.createElement("p");
  formula.className = "formula";
  formula.textContent = contribution.expression;
  block.append(formula);

  const terms = document.createElement("div");
  terms.className = "terms-list";
  if (contribution.terms.length === 0) {
    const empty = document.createElement("span");
    empty.className = "term-chip empty-note";
    empty.textContent = "No contributions for this channel at the selected output position.";
    terms.append(empty);
  } else {
    contribution.terms.forEach((term) => {
      const chip = document.createElement("span");
      chip.className = "term-chip";
      if (entry.mode === "conv") {
        chip.textContent =
          `${term.input_index ? `x[${term.input_index.join(",")}]` : "padding"}=` +
          `${formatNumber(term.input_value)} x ` +
          `w[${term.kernel_index.join(",")}]=${formatNumber(term.kernel_value)} ` +
          `= ${formatNumber(term.product)}`;
      } else {
        chip.textContent =
          `x[${term.input_index.join(",")}]=${formatNumber(term.input_value)} x ` +
          `w[${term.kernel_index.join(",")}]=${formatNumber(term.kernel_value)} ` +
          `= ${formatNumber(term.product)}`;
      }
      terms.append(chip);
    });
  }
  block.append(terms);

  return block;
}

function renderTrace(result) {
  elements.traceRender.replaceChildren();

  result.trace.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "trace-card";
    card.id = `trace-${entry.full_output_index.join("-")}`;

    const header = document.createElement("div");
    header.className = "trace-card-header";

    const title = document.createElement("h3");
    title.textContent = traceTitle(entry);

    const total = document.createElement("span");
    total.className = "status-pill";
    total.textContent = `= ${formatNumber(entry.output_value)}`;
    header.append(title, total);

    const meta = document.createElement("p");
    meta.className = "trace-meta";
    meta.textContent =
      `group ${entry.group} | pre-bias sum ${formatNumber(entry.pre_bias_value)} | ` +
      `bias ${entry.bias_value === null ? "off" : formatNumber(entry.bias_value)}`;

    const fullFormula = document.createElement("p");
    fullFormula.className = "formula";
    fullFormula.textContent = `${entry.expression} = ${formatNumber(entry.output_value)}`;

    card.append(header, meta, fullFormula);
    entry.channel_contributions.forEach((contribution) => {
      card.append(renderTraceContribution(result, entry, contribution));
    });

    elements.traceRender.append(card);
  });
}

function renderOutput(result) {
  elements.resultSummary.textContent =
    `${result.semantics.operation} with input tensor ${describeShape(
      result.input_tensor_shape,
    )}, weight tensor ${describeShape(result.weight_shape)}, and output tensor ${describeShape(
      result.output_tensor_shape,
    )}. ${result.semantics.note}`;

  const semanticPieces = [
    `stride ${describeShape(result.semantics.stride)}`,
    `padding ${describeShape(result.semantics.padding)}`,
    `dilation ${describeShape(result.semantics.dilation)}`,
    `groups ${result.semantics.groups}`,
  ];
  if (result.operation === "conv_transpose") {
    semanticPieces.push(`output padding ${describeShape(result.semantics.output_padding)}`);
  }
  semanticPieces.push(result.semantics.bias_enabled ? "bias on" : "bias off");
  elements.semanticsPill.textContent = semanticPieces.join(" | ");

  renderOutputArrays(result);
  renderOutputExplorer(result);
}

function renderAnimationFrame() {
  if (!state.latestResult || state.animationFrames.length === 0) {
    return;
  }

  const frame = state.animationFrames[state.animationIndex];
  const result = state.latestResult;
  elements.animationPanel.classList.remove("hidden");
  elements.animationStatus.textContent =
    `Frame ${state.animationIndex + 1} / ${state.animationFrames.length}`;
  elements.animationCaption.textContent =
    frame.mode === "conv"
      ? `The highlighted input samples and weight slices produce output[${frame.full_output_index.join(
          ", ",
        )}].`
      : `The highlighted source positions and active weights contribute to output[${frame.full_output_index.join(
          ", ",
        )}].`;

  elements.animationInputRender.replaceChildren();
  elements.animationKernelRender.replaceChildren();
  elements.animationOutputRender.replaceChildren();

  frame.channel_contributions.forEach((contribution) => {
    const inputCard = document.createElement("div");
    inputCard.className = "output-slice";
    const inputHeading = document.createElement("h3");
    inputHeading.textContent = `Input channel ${contribution.input_channel}`;
    inputCard.append(inputHeading);
    appendSpatialPreview(
      inputCard,
      result.input_values[contribution.input_channel],
      result.input_shape,
      "Highlighted positions",
      buildHighlightSet(
        contribution.terms
          .filter((term) => term.input_index !== null)
          .map((term) => term.input_index),
      ),
    );
    elements.animationInputRender.append(inputCard);

    const kernelCard = document.createElement("div");
    kernelCard.className = "output-slice";
    const kernelHeading = document.createElement("h3");
    kernelHeading.textContent = `Weight slice from input ${contribution.input_channel}`;
    kernelCard.append(kernelHeading);
    appendSpatialPreview(
      kernelCard,
      contribution.kernel_values,
      result.kernel_shape,
      "Active weights",
      buildHighlightSet(contribution.terms.map((term) => term.kernel_index)),
    );
    elements.animationKernelRender.append(kernelCard);
  });

  const outputHighlight = buildHighlightSet([frame.output_index]);
  const outputCard = document.createElement("div");
  outputCard.className = "output-slice";
  const outputHeading = document.createElement("h3");
  outputHeading.textContent = `Output channel ${frame.output_channel}`;
  outputCard.append(outputHeading);
  appendSpatialPreview(
    outputCard,
    result.output_values[frame.output_channel],
    result.output_spatial_shape,
    "Current target",
    null,
    outputHighlight,
  );
  elements.animationOutputRender.append(outputCard);
}

function buildAnimationFrames() {
  if (!state.latestResult) {
    return;
  }
  stopAnimationPlayback();
  state.animationFrames = state.latestResult.trace;
  state.animationIndex = 0;
  renderAnimationFrame();
}

function playAnimation() {
  if (state.animationFrames.length === 0) {
    return;
  }
  if (state.animationPlaying) {
    stopAnimationPlayback();
    return;
  }

  state.animationPlaying = true;
  elements.playAnimationButton.textContent = "Pause";
  state.animationTimer = window.setInterval(() => {
    state.animationIndex = (state.animationIndex + 1) % state.animationFrames.length;
    renderAnimationFrame();
  }, 1200);
}

async function computeConvolution() {
  const payload = {
    dimensions: state.dimensions,
    operation: state.operation,
    input_shape: state.inputShape,
    kernel_shape: state.kernelShape,
    input_channels: state.inputChannels,
    output_channels: state.outputChannels,
    groups: state.groups,
    stride: state.stride,
    padding: state.padding,
    dilation: state.dilation,
    output_padding: state.outputPadding,
    bias_enabled: state.biasEnabled,
    input_values: state.inputValues,
    kernel_values: state.kernelValues,
    bias_values: state.biasValues,
  };

  elements.resultError.classList.add("hidden");
  elements.calculateButton.disabled = true;
  elements.calculateButton.textContent = "Calculating...";

  try {
    const response = await fetch("/api/compute", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Unable to compute convolution.");
    }

    state.latestResult = data;
    renderOutput(data);
    renderTrace(data);
    elements.generateAnimationButton.disabled = false;
  } catch (error) {
    state.latestResult = null;
    elements.outputArrayRender.replaceChildren();
    elements.outputCardRender.replaceChildren();
    elements.traceRender.replaceChildren();
    elements.resultSummary.textContent = "";
    elements.resultError.textContent = error.message;
    elements.resultError.classList.remove("hidden");
    elements.generateAnimationButton.disabled = true;
    clearAnimation();
  } finally {
    elements.calculateButton.disabled = false;
    elements.calculateButton.textContent = "Calculate";
  }
}

function queueAutoCompute() {
  invalidateComputedViews();
  if (state.pendingTimer) {
    window.clearTimeout(state.pendingTimer);
  }
  state.pendingTimer = window.setTimeout(() => {
    computeConvolution();
  }, 260);
}

function randomizeCurrentValues() {
  state.inputValues = createFilledTensor(getInputTensorShape(), () => randomValue());
  state.kernelValues = createFilledTensor(getWeightShape(), () => randomValue());
  state.biasValues = resizeVector([], state.outputChannels, 0).map(() => randomValue());
  renderState();
  queueAutoCompute();
}

elements.dimensions.addEventListener("change", () => {
  loadPreset(Number(elements.dimensions.value), state.operation);
  renderState();
  queueAutoCompute();
});

elements.operation.addEventListener("change", () => {
  loadPreset(state.dimensions, elements.operation.value);
  renderState();
  queueAutoCompute();
});

elements.inputChannels.addEventListener("input", () => {
  const snapshot = snapshotStateShapes();
  const parsed = Number.parseInt(elements.inputChannels.value || "1", 10);
  state.inputChannels = Math.max(1, Number.isFinite(parsed) ? parsed : 1);
  normalizeStateFromSnapshot(snapshot);
  renderState();
  queueAutoCompute();
});

elements.outputChannels.addEventListener("input", () => {
  const snapshot = snapshotStateShapes();
  const parsed = Number.parseInt(elements.outputChannels.value || "1", 10);
  state.outputChannels = Math.max(1, Number.isFinite(parsed) ? parsed : 1);
  normalizeStateFromSnapshot(snapshot);
  renderState();
  queueAutoCompute();
});

elements.groups.addEventListener("change", () => {
  const snapshot = snapshotStateShapes();
  state.groups = Number.parseInt(elements.groups.value, 10);
  normalizeStateFromSnapshot(snapshot);
  renderState();
  queueAutoCompute();
});

elements.biasEnabled.addEventListener("change", () => {
  state.biasEnabled = elements.biasEnabled.checked;
  renderState();
  queueAutoCompute();
});

elements.calculateButton.addEventListener("click", () => {
  computeConvolution();
});

elements.generateAnimationButton.addEventListener("click", () => {
  buildAnimationFrames();
});

elements.prevFrameButton.addEventListener("click", () => {
  if (state.animationFrames.length === 0) {
    return;
  }
  stopAnimationPlayback();
  state.animationIndex =
    (state.animationIndex - 1 + state.animationFrames.length) %
    state.animationFrames.length;
  renderAnimationFrame();
});

elements.nextFrameButton.addEventListener("click", () => {
  if (state.animationFrames.length === 0) {
    return;
  }
  stopAnimationPlayback();
  state.animationIndex = (state.animationIndex + 1) % state.animationFrames.length;
  renderAnimationFrame();
});

elements.playAnimationButton.addEventListener("click", () => {
  playAnimation();
});

elements.exampleButton.addEventListener("click", () => {
  loadPreset();
  renderState();
  queueAutoCompute();
});

elements.randomizeButton.addEventListener("click", () => {
  randomizeCurrentValues();
});

renderState();
computeConvolution();
