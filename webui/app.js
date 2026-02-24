const state = {
  nodes: [],
  edges: [],
  selectedId: null,
  connectMode: false,
  connectFrom: null,
  pan: { x: 0, y: 0, scale: 1 },
  clickTimer: null,
  draggingNodeId: null,
  draggingOffset: null,
  draggingMoved: false,
  draggingJustEnded: false,
  meta: { imports: [] },
};

const nodeTypeOrder = [
  "data",
  "filter",
  "transform",
  "loader",
  "image_tower",
  "metadata_tower",
  "bridge",
  "classifier",
];

const typeMeta = {
  data: { color: "#2f5d62", label: "Data" },
  filter: { color: "#8d6b94", label: "Filter" },
  transform: { color: "#f6c453", label: "Transform" },
  loader: { color: "#c84630", label: "Loader" },
  image_tower: { color: "#1f4e79", label: "Tower" },
  metadata_tower: { color: "#6b4226", label: "Tower" },
  bridge: { color: "#2a2a72", label: "Bridge" },
  classifier: { color: "#5a3e2b", label: "Classifier" },
};

const svg = document.getElementById("graph");
const jsonView = document.getElementById("json-view");
const nodeCount = document.getElementById("node-count");
const edgeCount = document.getElementById("edge-count");
const selectedNode = document.getElementById("selected-node");
const nodeLabelInput = document.getElementById("node-label");
const nodeSlotInput = document.getElementById("node-slot");
const nodeOutputInput = document.getElementById("node-output");
const selectionGeneric = document.getElementById("selection-generic");
const selectionLoader = document.getElementById("selection-loader");
const selectionFilter = document.getElementById("selection-filter");
const selectionTransform = document.getElementById("selection-transform");
const selectionTower = document.getElementById("selection-tower");
const selectionBridge = document.getElementById("selection-bridge");
const selectionClassifier = document.getElementById("selection-classifier");
const selectionData = document.getElementById("selection-data");
const loaderInputType = document.getElementById("loader-input-type");
const loaderInputIndex = document.getElementById("loader-input-index");
const loaderOutputType = document.getElementById("loader-output-type");
const dataOutputType = document.getElementById("data-output-type");
const filterInputType = document.getElementById("filter-input-type");
const filterOutputType = document.getElementById("filter-output-type");
const filterType = document.getElementById("filter-type");
const filterRegexFields = document.getElementById("filter-regex-fields");
const filterRegexPattern = document.getElementById("filter-regex-pattern");
const filterColumnFields = document.getElementById("filter-column-fields");
const filterColumnName = document.getElementById("filter-column-name");
const filterColumnOperator = document.getElementById("filter-column-operator");
const filterColumnValue = document.getElementById("filter-column-value");
const transformType = document.getElementById("transform-type");
const transformRoiFields = document.getElementById("transform-roi-fields");
const transformRoiMaskSource = document.getElementById("transform-roi-mask-source");
const transformRoiScale = document.getElementById("transform-roi-scale");
const transformRoiTarget = document.getElementById("transform-roi-target");
const transformRoiFallback = document.getElementById("transform-roi-fallback");
const transformCenterFields = document.getElementById("transform-center-fields");
const transformCenterSize = document.getElementById("transform-center-size");
const transformJitterFields = document.getElementById("transform-jitter-fields");
const transformJitterHFlip = document.getElementById("transform-jitter-hflip");
const transformJitterVFlip = document.getElementById("transform-jitter-vflip");
const transformJitterRotation = document.getElementById("transform-jitter-rotation");
const transformJitterColorEnabled = document.getElementById("transform-jitter-color-enabled");
const transformJitterColor = document.getElementById("transform-jitter-color");
const transformResizeFields = document.getElementById("transform-resize-fields");
const transformResizeSize = document.getElementById("transform-resize-size");
const towerType = document.getElementById("tower-type");
const towerImageFields = document.getElementById("tower-image-fields");
const towerBackbone = document.getElementById("tower-backbone");
const towerFreezeRatio = document.getElementById("tower-freeze-ratio");
const towerAugment = document.getElementById("tower-augment");
const towerGeometryDim = document.getElementById("tower-geometry-dim");
const towerUseSe = document.getElementById("tower-use-se");
const towerSeReduction = document.getElementById("tower-se-reduction");
const towerSePreNorm = document.getElementById("tower-se-pre-norm");
const towerMdFields = document.getElementById("tower-md-fields");
const towerMdHidden = document.getElementById("tower-md-hidden");
const towerMdDropout = document.getElementById("tower-md-dropout");
const towerMdUseSe = document.getElementById("tower-md-use-se");
const towerMdSeReduction = document.getElementById("tower-md-se-reduction");
const towerMdSePreNorm = document.getElementById("tower-md-se-pre-norm");
const towerMdFreezeRatio = document.getElementById("tower-md-freeze-ratio");
const bridgeMethod = document.getElementById("bridge-method");
const bridgeFusionFields = document.getElementById("bridge-fusion-fields");
const bridgeFusionDim = document.getElementById("bridge-fusion-dim");
const bridgeUseSe = document.getElementById("bridge-use-se");
const bridgeSeReduction = document.getElementById("bridge-se-reduction");
const bridgeSePreNorm = document.getElementById("bridge-se-pre-norm");
const dataBrowseButton = document.getElementById("data-browse");
const dataSourceStatus = document.getElementById("data-source-status");
const duplicateDataButton = document.getElementById("duplicate-data");
const modalOverlay = document.getElementById("modal-overlay");
const modalList = document.getElementById("modal-list");
const modalClose = document.getElementById("modal-close");
const modalSelectDir = document.getElementById("modal-select-dir");
const modalSelectFile = document.getElementById("modal-select-file");
const modalMode = document.getElementById("modal-mode");
const breadcrumb = document.getElementById("breadcrumb");
const previewOverlay = document.getElementById("preview-overlay");
const previewClose = document.getElementById("preview-close");
const previewBody = document.getElementById("preview-body");
const previewSubtitle = document.getElementById("preview-subtitle");
const importClassSelect = document.getElementById("import-class-select");
const importClassBtn = document.getElementById("import-class-btn");
const presetSelect = document.getElementById("preset-select");
const presetApplyBtn = document.getElementById("preset-apply");
const presetSaveBtn = document.getElementById("preset-save");
const connectButton = document.getElementById("connect-mode");
const toggleJsonButton = document.getElementById("toggle-json");

const PRESET_STORAGE_KEY = "hypertower_v2_presets";
const selectionState = { nodeIds: new Set(), edgeIndices: new Set() };
let selectionBox = null;

function newNodeId() {
  return `node_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
}

function createNodeTemplate(type) {
  const node = {
    id: newNodeId(),
    type,
    label: `${typeMeta[type]?.label || type}`,
    inputKey: "",
    outputKey: "",
    inputType: "image",
    inputIndex: "1",
    outputType: type === "data" ? "" : "image",
    source: null,
    selectedPath: null,
    filterType: type === "filter" ? "regex" : "",
    regexPattern: "",
    columnName: "",
    columnOperator: ">",
    columnValue: "",
    transformType: type === "transform" ? "center_crop" : "",
    roiMaskSource: "gt",
    roiScale: 2.5,
    roiTargetSize: 224,
    roiFallback: true,
    centerCropSize: 224,
    jitterHFlip: true,
    jitterVFlip: true,
    jitterRotation: 15,
    jitterColorEnabled: true,
    jitterColor: "0.1,0.1,0.1,0.05",
    resizeSize: 256,
  };
  if (type === "image_tower" || type === "metadata_tower") {
    node.towerType = type === "metadata_tower" ? "metadata" : "image";
    node.imageBackbone = "efficientnet_b0";
    node.imageFreezeRatio = 0.0;
    node.imageAugment = true;
    node.imageGeometryDim = 0;
    node.imageUseSe = false;
    node.imageSeReduction = 16;
    node.imageSePreNorm = true;
    node.mdHiddenDim = 128;
    node.mdDropout = 0.1;
    node.mdUseSe = false;
    node.mdSeReduction = 16;
    node.mdSePreNorm = true;
    node.mdFreezeRatio = 0.0;
  }
  if (type === "bridge") {
    node.bridgeMethod = "fusion";
    node.bridgeFusionDim = 256;
    node.bridgeUseSe = true;
    node.bridgeSeReduction = 16;
    node.bridgeSePreNorm = true;
  }
  return node;
}

function addNode(type) {
  const count = state.nodes.filter((n) => n.type === type).length + 1;
  const node = createNodeTemplate(type);
  node.label = `${typeMeta[type]?.label || type} ${count}`;
  state.nodes.push(node);
  selectNode(node.id);
  autoLayout();
  render();
  return node;
}

function addDataSourceNode({ label, outputType }) {
  const node = createNodeTemplate("data");
  node.label = label;
  node.inputType = "";
  node.inputIndex = "";
  node.outputType = outputType;
  node.sourceRef = null;
  state.nodes.push(node);
  return node;
}

async function importClassDefinition() {
  const selection = importClassSelect.value;
  if (!selection) return;
  if (selection === "PapilaData") {
    let payload = null;
    try {
      const res = await fetch("/api/import/papila");
      const parsed = await parseJsonResponse(res);
      payload = parsed.data;
      if (!res.ok || !payload) {
        throw new Error(payload?.error || parsed.text || "Import failed");
      }
    } catch (err) {
      alert(`Papila import failed: ${err.message}`);
      return;
    }

    const importId = `import_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    if (!state.meta) {
      state.meta = { imports: [] };
    }
    if (!Array.isArray(state.meta.imports)) {
      state.meta.imports = [];
    }
    state.meta.imports.push({
      id: importId,
      className: "PapilaData",
      params: {
        image_dir: "Papila/FundusImages",
        clinical_dir: "Papila/ClinicalData",
        label_col: payload.label_col || "Diagnosis",
        cat_cols: ["Gender", "Phakic/Pseudophakic"],
      },
    });

    const imageNode = addDataSourceNode({
      label: "Papila Images",
      outputType: "image",
    });
    imageNode.source = {
      mode: "directory",
      path: payload.image_dir,
      count: null,
    };
    imageNode.sourceRef = {
      importId,
      slot: "image_dir",
      mode: "directory",
    };

    const matrixNode = addDataSourceNode({
      label: "Papila Metadata",
      outputType: "matrix",
    });
    matrixNode.source = {
      mode: "dataframe",
      name: "clinical.df",
      rows: payload.df_rows,
      cols: payload.df_cols,
    };
    matrixNode.sourceRef = {
      importId,
      slot: "clinical_df",
      mode: "dataframe",
    };
    const resizeTransform = addNode("transform");
    resizeTransform.label = "Resize";
    resizeTransform.transformType = "resize";
    resizeTransform.resizeSize = 256;

    const centerTransform = addNode("transform");
    centerTransform.label = "Center Crop";
    centerTransform.transformType = "center_crop";
    centerTransform.centerCropSize = 224;

    const jitterTransform = addNode("transform");
    jitterTransform.label = "Jitter";
    jitterTransform.transformType = "jitter_bundle";
    jitterTransform.jitterHFlip = true;
    jitterTransform.jitterVFlip = true;
    jitterTransform.jitterRotation = 15;
    jitterTransform.jitterColorEnabled = true;
    jitterTransform.jitterColor = "0.1,0.1,0.1,0.05";

    const imageLoader = addNode("loader");
    imageLoader.label = "Image Loader";
    imageLoader.inputType = "image";
    imageLoader.inputIndex = "1";
    imageLoader.outputType = "image";
    imageLoader.inputKey = "image_1";
    imageLoader.outputKey = "image_1";

    const matrixLoader = addNode("loader");
    matrixLoader.label = "Metadata Loader";
    matrixLoader.inputType = "matrix";
    matrixLoader.inputIndex = "1";
    matrixLoader.outputType = "matrix";
    matrixLoader.inputKey = "matrix_1";
    matrixLoader.outputKey = "matrix_1";

    const imageTower = addNode("image_tower");
    imageTower.label = "Tower 1";
    imageTower.towerType = "image";
    imageTower.imageBackbone = "efficientnet_b0";
    imageTower.imageFreezeRatio = 0.0;
    imageTower.imageAugment = true;
    imageTower.imageGeometryDim = 0;
    imageTower.imageUseSe = false;
    imageTower.imageSeReduction = 16;
    imageTower.imageSePreNorm = true;

    const metadataTower = addNode("metadata_tower");
    metadataTower.label = "Tower 2";
    metadataTower.towerType = "metadata";
    metadataTower.mdHiddenDim = 128;
    metadataTower.mdDropout = 0.1;
    metadataTower.mdUseSe = false;
    metadataTower.mdSeReduction = 16;
    metadataTower.mdSePreNorm = true;
    metadataTower.mdFreezeRatio = 0.0;

    const bridgeNode = addNode("bridge");
    bridgeNode.label = "Bridge";
    bridgeNode.bridgeMethod = "fusion";
    bridgeNode.bridgeFusionDim = 256;
    bridgeNode.bridgeUseSe = true;
    bridgeNode.bridgeSeReduction = 16;
    bridgeNode.bridgeSePreNorm = true;

    const classifierNode = addNode("classifier");
    classifierNode.label = "Classifier";

    state.edges.push({ from: imageNode.id, to: resizeTransform.id });
    state.edges.push({ from: resizeTransform.id, to: centerTransform.id });
    state.edges.push({ from: centerTransform.id, to: jitterTransform.id });
    state.edges.push({ from: jitterTransform.id, to: imageLoader.id });
    state.edges.push({ from: matrixNode.id, to: matrixLoader.id });
    state.edges.push({ from: imageLoader.id, to: imageTower.id });
    state.edges.push({ from: matrixLoader.id, to: metadataTower.id });
    state.edges.push({ from: imageTower.id, to: bridgeNode.id });
    state.edges.push({ from: metadataTower.id, to: bridgeNode.id });
    state.edges.push({ from: bridgeNode.id, to: classifierNode.id });

    autoLayout();
    selectNode(imageNode.id);
    render();
  }
}

function selectNode(id) {
  state.selectedId = id;
  if (id) {
    setSelection([id]);
  } else {
    setSelection([]);
  }
  const node = state.nodes.find((n) => n.id === id);
  if (node) {
    selectedNode.textContent = node.label;
    selectedNode.classList.remove("muted");
    nodeLabelInput.value = node.label;
    nodeSlotInput.value = node.inputKey || "";
    nodeOutputInput.value = node.outputKey || "";
    selectionGeneric.classList.toggle("hidden", node.type === "loader" || node.type === "data");
    selectionLoader.classList.toggle("hidden", node.type !== "loader");
    selectionFilter.classList.toggle("hidden", node.type !== "filter");
    selectionTransform.classList.toggle("hidden", node.type !== "transform");
    const isTower = node.type === "image_tower" || node.type === "metadata_tower";
    selectionTower.classList.toggle("hidden", !isTower);
    selectionBridge.classList.toggle("hidden", node.type !== "bridge");
    selectionClassifier.classList.toggle("hidden", node.type !== "classifier");
    selectionData.classList.toggle("hidden", node.type !== "data");

    if (node.type === "loader") {
      loaderInputType.value = node.inputType || "image";
      loaderInputIndex.value = node.inputIndex || "1";
      loaderOutputType.value = node.outputType || loaderInputType.value;
    }
    if (node.type === "filter") {
      filterInputType.value = node.inputType || "image";
      filterOutputType.value = node.outputType || filterInputType.value;
      filterType.value = node.filterType || "regex";
      filterRegexPattern.value = node.regexPattern || "";
      filterColumnName.value = node.columnName || "";
      filterColumnOperator.value = node.columnOperator || ">";
      filterColumnValue.value = node.columnValue || "";
      updateFilterFieldVisibility(filterType.value);
    }
    if (node.type === "transform") {
      transformType.value = node.transformType || "center_crop";
      updateTransformFieldVisibility(transformType.value);
      transformRoiMaskSource.value = node.roiMaskSource || "gt";
      transformRoiScale.value =
        node.roiScale === null || node.roiScale === undefined ? "" : node.roiScale;
      transformRoiTarget.value =
        node.roiTargetSize === null || node.roiTargetSize === undefined ? "" : node.roiTargetSize;
      transformRoiFallback.checked = node.roiFallback !== false;
      transformCenterSize.value =
        node.centerCropSize === null || node.centerCropSize === undefined
          ? ""
          : node.centerCropSize;
      transformJitterHFlip.checked = node.jitterHFlip !== false;
      transformJitterVFlip.checked = node.jitterVFlip !== false;
      transformJitterRotation.value =
        node.jitterRotation === null || node.jitterRotation === undefined
          ? ""
          : node.jitterRotation;
      transformJitterColorEnabled.checked = node.jitterColorEnabled !== false;
      transformJitterColor.value = node.jitterColor || "";
      transformResizeSize.value =
        node.resizeSize === null || node.resizeSize === undefined ? "" : node.resizeSize;
    }
    if (node.type === "bridge") {
      bridgeMethod.value = node.bridgeMethod || "fusion";
      updateBridgeFieldVisibility(bridgeMethod.value);
      bridgeFusionDim.value =
        node.bridgeFusionDim === null || node.bridgeFusionDim === undefined
          ? ""
          : node.bridgeFusionDim;
      bridgeUseSe.checked = node.bridgeUseSe !== false;
      bridgeSeReduction.value =
        node.bridgeSeReduction === null || node.bridgeSeReduction === undefined
          ? ""
          : node.bridgeSeReduction;
      bridgeSePreNorm.checked = node.bridgeSePreNorm !== false;
    }
    if (node.type === "image_tower" || node.type === "metadata_tower") {
      const towerKind = node.type === "metadata_tower" ? "metadata" : "image";
      towerType.value = node.towerType || towerKind;
      updateTowerFieldVisibility(towerType.value);
      towerBackbone.value = node.imageBackbone || "efficientnet_b0";
      towerFreezeRatio.value =
        node.imageFreezeRatio === null || node.imageFreezeRatio === undefined
          ? ""
          : node.imageFreezeRatio;
      towerAugment.checked = node.imageAugment !== false;
      towerGeometryDim.value =
        node.imageGeometryDim === null || node.imageGeometryDim === undefined
          ? ""
          : node.imageGeometryDim;
      towerUseSe.checked = node.imageUseSe === true;
      towerSeReduction.value =
        node.imageSeReduction === null || node.imageSeReduction === undefined
          ? ""
          : node.imageSeReduction;
      towerSePreNorm.checked = node.imageSePreNorm !== false;
      towerMdHidden.value =
        node.mdHiddenDim === null || node.mdHiddenDim === undefined ? "" : node.mdHiddenDim;
      towerMdDropout.value =
        node.mdDropout === null || node.mdDropout === undefined ? "" : node.mdDropout;
      towerMdUseSe.checked = node.mdUseSe === true;
      towerMdSeReduction.value =
        node.mdSeReduction === null || node.mdSeReduction === undefined
          ? ""
          : node.mdSeReduction;
      towerMdSePreNorm.checked = node.mdSePreNorm !== false;
      towerMdFreezeRatio.value =
        node.mdFreezeRatio === null || node.mdFreezeRatio === undefined
          ? ""
          : node.mdFreezeRatio;
    }
    if (node.type === "data") {
      dataOutputType.value = node.outputType || "";
      updateDataBrowseVisibility(node.outputType || "");
      updateDataSourceStatus(node);
    }
  } else {
    clearSelectionUI();
  }
}

function clearSelectionUI() {
  selectedNode.textContent = "None";
  selectedNode.classList.add("muted");
  nodeLabelInput.value = "";
  nodeSlotInput.value = "";
  nodeOutputInput.value = "";
  selectionGeneric.classList.remove("hidden");
  selectionLoader.classList.add("hidden");
  selectionFilter.classList.add("hidden");
  selectionTransform.classList.add("hidden");
  selectionTower.classList.add("hidden");
  selectionBridge.classList.add("hidden");
  selectionClassifier.classList.add("hidden");
  selectionData.classList.add("hidden");
}

function applyNodeEdits() {
  const node = state.nodes.find((n) => n.id === state.selectedId);
  if (!node) return;
  node.label = nodeLabelInput.value.trim() || node.label;
  node.inputKey = nodeSlotInput.value.trim();
  node.outputKey = nodeOutputInput.value.trim();
  if (node.type === "loader") {
    node.inputType = loaderInputType.value;
    node.inputIndex = loaderInputIndex.value;
    node.outputType = node.inputType;
  }
  if (node.type === "filter") {
    node.inputType = filterInputType.value;
    node.outputType = node.inputType;
    node.filterType = filterType.value;
    node.regexPattern = filterRegexPattern.value.trim();
    node.columnName = filterColumnName.value.trim();
    node.columnOperator = filterColumnOperator.value;
    node.columnValue = filterColumnValue.value.trim();
    filterOutputType.value = node.outputType;
  }
  if (node.type === "transform") {
    node.transformType = transformType.value;
    node.roiMaskSource = transformRoiMaskSource.value;
    node.roiScale = transformRoiScale.value === "" ? null : Number(transformRoiScale.value);
    node.roiTargetSize =
      transformRoiTarget.value === "" ? null : Number(transformRoiTarget.value);
    node.roiFallback = transformRoiFallback.checked;
    node.centerCropSize =
      transformCenterSize.value === "" ? null : Number(transformCenterSize.value);
    node.jitterHFlip = transformJitterHFlip.checked;
    node.jitterVFlip = transformJitterVFlip.checked;
    node.jitterRotation =
      transformJitterRotation.value === "" ? null : Number(transformJitterRotation.value);
    node.jitterColorEnabled = transformJitterColorEnabled.checked;
    node.jitterColor = transformJitterColor.value.trim();
    node.resizeSize =
      transformResizeSize.value === "" ? null : Number(transformResizeSize.value);
  }
  if (node.type === "bridge") {
    node.bridgeMethod = bridgeMethod.value;
    node.bridgeFusionDim =
      bridgeFusionDim.value === "" ? null : Number(bridgeFusionDim.value);
    node.bridgeUseSe = bridgeUseSe.checked;
    node.bridgeSeReduction =
      bridgeSeReduction.value === "" ? null : Number(bridgeSeReduction.value);
    node.bridgeSePreNorm = bridgeSePreNorm.checked;
  }
  if (node.type === "image_tower" || node.type === "metadata_tower") {
    const desiredType = towerType.value === "metadata" ? "metadata_tower" : "image_tower";
    if (node.type !== desiredType) {
      const oldType = node.type;
      node.type = desiredType;
      const oldLabelBase = typeMeta[oldType]?.label;
      const newLabelBase = typeMeta[desiredType]?.label;
      if (oldLabelBase && newLabelBase && node.label.startsWith(oldLabelBase)) {
        node.label = node.label.replace(oldLabelBase, newLabelBase);
      }
    }
    node.towerType = towerType.value;
    node.imageBackbone = towerBackbone.value;
    node.imageFreezeRatio =
      towerFreezeRatio.value === "" ? null : Number(towerFreezeRatio.value);
    node.imageAugment = towerAugment.checked;
    node.imageGeometryDim =
      towerGeometryDim.value === "" ? null : Number(towerGeometryDim.value);
    node.imageUseSe = towerUseSe.checked;
    node.imageSeReduction =
      towerSeReduction.value === "" ? null : Number(towerSeReduction.value);
    node.imageSePreNorm = towerSePreNorm.checked;
    node.mdHiddenDim = towerMdHidden.value === "" ? null : Number(towerMdHidden.value);
    node.mdDropout = towerMdDropout.value === "" ? null : Number(towerMdDropout.value);
    node.mdUseSe = towerMdUseSe.checked;
    node.mdSeReduction =
      towerMdSeReduction.value === "" ? null : Number(towerMdSeReduction.value);
    node.mdSePreNorm = towerMdSePreNorm.checked;
    node.mdFreezeRatio =
      towerMdFreezeRatio.value === "" ? null : Number(towerMdFreezeRatio.value);
  }
  if (node.type === "data") {
    node.outputType = dataOutputType.value;
    updateDataSourceStatus(node);
  }
  render();
}

function deleteSelected() {
  if (!state.selectedId) return;
  state.nodes = state.nodes.filter((n) => n.id !== state.selectedId);
  state.edges = state.edges.filter(
    (e) => e.from !== state.selectedId && e.to !== state.selectedId
  );
  state.selectedId = null;
  selectNode(null);
  render();
}

function toggleConnectMode() {
  state.connectMode = !state.connectMode;
  state.connectFrom = null;
  connectButton.textContent = state.connectMode ? "Connecting..." : "Connect";
  connectButton.classList.toggle("danger", state.connectMode);
}

function connectNodes(fromId, toId) {
  if (!fromId || !toId || fromId === toId) return;
  const exists = state.edges.some((e) => e.from === fromId && e.to === toId);
  if (!exists) {
    const toNode = state.nodes.find((n) => n.id === toId);
    if (toNode && toNode.type === "bridge") {
      const incoming = state.edges.filter((e) => e.to === toId);
      if (incoming.length >= 2) {
        alert("Bridge nodes require exactly two inputs. Remove an edge first.");
        return;
      }
    }
    state.edges.push({ from: fromId, to: toId });
    if (toNode && (toNode.type === "image_tower" || toNode.type === "metadata_tower")) {
      autoSetTowerType(toNode);
      if (state.selectedId === toNode.id) {
        selectNode(toNode.id);
      }
    }
  }
}

function removeSelectedEdges() {
  if (!state.selectedId) return;
  state.edges = state.edges.filter(
    (e) => e.from !== state.selectedId && e.to !== state.selectedId
  );
  const node = state.nodes.find((n) => n.id === state.selectedId);
  if (node && (node.type === "image_tower" || node.type === "metadata_tower")) {
    autoSetTowerType(node);
  }
  render();
}

function autoLayout() {
  const rows = {};
  nodeTypeOrder.forEach((type, idx) => {
    rows[type] = { y: 760 - idx * 130, nodes: [] };
  });
  state.nodes.forEach((node) => {
    if (!rows[node.type]) {
      rows[node.type] = { y: 760 - nodeTypeOrder.length * 130, nodes: [] };
    }
    rows[node.type].nodes.push(node);
  });
  Object.values(rows).forEach((row) => {
    row.nodes.forEach((node, index) => {
      node.x = 160 + index * 180;
      node.y = row.y;
    });
  });
}

function updateDataBrowseVisibility(outputType) {
  dataBrowseButton.disabled = outputType === "";
  modalSelectDir.disabled = outputType !== "image";
  modalSelectFile.disabled = outputType !== "matrix";
}

function updateFilterFieldVisibility(type) {
  filterRegexFields.classList.toggle("hidden", type !== "regex");
  filterColumnFields.classList.toggle("hidden", type !== "column");
}

function updateTransformFieldVisibility(type) {
  transformRoiFields.classList.toggle("hidden", type !== "roi_crop");
  transformCenterFields.classList.toggle("hidden", type !== "center_crop");
  transformJitterFields.classList.toggle("hidden", type !== "jitter_bundle");
  transformResizeFields.classList.toggle("hidden", type !== "resize");
}

function updateTowerFieldVisibility(type) {
  towerImageFields.classList.toggle("hidden", type !== "image");
  towerMdFields.classList.toggle("hidden", type !== "metadata");
}

function updateBridgeFieldVisibility(method) {
  bridgeFusionFields.classList.toggle("hidden", method !== "fusion");
}

function setSelection(nodeIds) {
  selectionState.nodeIds = new Set(nodeIds);
  selectionState.edgeIndices = new Set();
  state.edges.forEach((edge, idx) => {
    if (selectionState.nodeIds.has(edge.from) && selectionState.nodeIds.has(edge.to)) {
      selectionState.edgeIndices.add(idx);
    }
  });
}

function autoSetTowerType(towerNode) {
  if (!towerNode) return;
  const incoming = state.edges.filter((e) => e.to === towerNode.id);
  const fromNodes = incoming
    .map((edge) => state.nodes.find((n) => n.id === edge.from))
    .filter(Boolean);
  const loader = fromNodes.find((n) => n.type === "loader");
  if (!loader) return;
  const desired = loader.inputType === "matrix" ? "metadata" : "image";
  if (towerNode.towerType !== desired) {
    towerNode.towerType = desired;
    towerNode.type = desired === "metadata" ? "metadata_tower" : "image_tower";
    const labelBase = typeMeta[towerNode.type]?.label || "Tower";
    if (!towerNode.label || towerNode.label.startsWith("Tower")) {
      const count = state.nodes.filter((n) => n.type === towerNode.type).length;
      towerNode.label = `${labelBase} ${count + 1}`;
    } else if (towerNode.label.startsWith("Tower")) {
      towerNode.label = towerNode.label.replace(/^Tower/, labelBase);
    }
  }
}

function resolveUpstreamData(startId) {
  const visited = new Set();
  const filters = [];
  let currentId = startId;
  let dataNode = null;
  let ambiguous = false;
  while (currentId) {
    if (visited.has(currentId)) break;
    visited.add(currentId);
    const incoming = state.edges.filter((e) => e.to === currentId);
    if (!incoming.length) break;
    if (incoming.length > 1) ambiguous = true;
    const fromId = incoming[0].from;
    const fromNode = state.nodes.find((n) => n.id === fromId);
    if (!fromNode) break;
    if (fromNode.type === "filter") {
      filters.push(fromNode);
      currentId = fromNode.id;
      continue;
    }
    if (fromNode.type === "data") {
      dataNode = fromNode;
      break;
    }
    currentId = fromNode.id;
  }
  return { dataNode, filters, ambiguous };
}

function formatFilterLabel(filter) {
  if (filter.filterType === "column") {
    const name = filter.columnName || "?";
    const op = filter.columnOperator || "?";
    const value = filter.columnValue || "?";
    return `col:${name} ${op} ${value}`;
  }
  const pattern = filter.regexPattern || "";
  return `regex:${pattern || "?"}`;
}

function summarizeFilters(filters) {
  if (!filters.length) return "";
  return filters.map(formatFilterLabel).join(", ");
}

function compileRegex(pattern) {
  if (!pattern) return { regex: null, error: null };
  try {
    return { regex: new RegExp(pattern), error: null };
  } catch (err) {
    return { regex: null, error: err.message || "invalid regex" };
  }
}

function applyRegexFilters(entries, filters) {
  let filtered = entries;
  const warnings = [];
  filters.forEach((filter) => {
    if (filter.filterType !== "regex") return;
    const pattern = (filter.regexPattern || "").trim();
    if (!pattern) return;
    const { regex, error } = compileRegex(pattern);
    if (error || !regex) {
      warnings.push(`Invalid regex "${pattern}": ${error || "invalid"}`);
      return;
    }
    filtered = filtered.filter((entry) => regex.test(entry.name));
  });
  return { filtered, warnings };
}

function compareCell(cell, rawValue, operator) {
  const cellStr = cell == null ? "" : String(cell).trim();
  const valueStr = rawValue == null ? "" : String(rawValue).trim();
  const cellNorm = cellStr.toLowerCase();
  const valueNorm = valueStr.toLowerCase();
  if (operator === "=") return cellNorm === valueNorm;
  if (operator === "!=") return cellNorm !== valueNorm;
  const cellNum = Number.parseFloat(cellStr);
  const valueNum = Number.parseFloat(valueStr);
  if (!Number.isFinite(cellNum) || !Number.isFinite(valueNum)) return false;
  switch (operator) {
    case ">":
      return cellNum > valueNum;
    case ">=":
      return cellNum >= valueNum;
    case "<":
      return cellNum < valueNum;
    case "<=":
      return cellNum <= valueNum;
    default:
      return false;
  }
}

function applyColumnFilters(header, rows, filters) {
  let filteredRows = rows;
  const warnings = [];
  filters.forEach((filter) => {
    if (filter.filterType !== "column") return;
    const columnName = (filter.columnName || "").trim();
    if (!columnName) {
      warnings.push("Column filter missing column name.");
      return;
    }
    let colIndex = header.indexOf(columnName);
    if (colIndex === -1) {
      const lower = columnName.toLowerCase();
      const matches = header
        .map((col, idx) => ({ col, idx }))
        .filter((item) => String(item.col).toLowerCase() === lower);
      if (matches.length) {
        colIndex = matches[0].idx;
        if (matches.length > 1) {
          warnings.push(`Column "${columnName}" matched multiple headers; using "${matches[0].col}".`);
        }
      }
    }
    if (colIndex === -1) {
      warnings.push(`Column "${columnName}" not found.`);
      return;
    }
    const rawValue = filter.columnValue;
    if (rawValue === "" || rawValue == null) {
      warnings.push(`Column filter "${columnName}" missing value.`);
      return;
    }
    const op = filter.columnOperator || "=";
    filteredRows = filteredRows.filter((row) =>
      compareCell(row[colIndex], rawValue, op)
    );
  });
  return { filteredRows, warnings };
}

function renderPreviewWarnings(warnings) {
  if (!warnings.length) return null;
  const wrap = document.createElement("div");
  warnings.forEach((msg) => {
    const line = document.createElement("div");
    line.className = "hint";
    line.textContent = msg;
    wrap.appendChild(line);
  });
  return wrap;
}

function renderPreviewCount(label, shown, total, truncated, note = "") {
  const line = document.createElement("div");
  line.className = "hint";
  const truncation = truncated ? " (showing first 200)" : "";
  const suffix = note ? ` ${note}` : "";
  line.textContent = `${label}: ${shown} / ${total}${truncation}${suffix}`;
  return line;
}

function updateDataSourceStatus(node) {
  if (!node || !node.source) {
    dataSourceStatus.textContent = "No source selected.";
    return;
  }
  if (node.source.mode === "directory") {
    const count = node.source.count === null ? "?" : node.source.count;
    dataSourceStatus.textContent = `Directory: ${node.source.path} (${count} files)`;
  } else if (node.source.mode === "file") {
    dataSourceStatus.textContent = `File: ${node.source.path} (${node.source.size} bytes)`;
  } else if (node.source.mode === "dataframe") {
    dataSourceStatus.textContent = `DataFrame: ${node.source.name} (${node.source.rows}x${node.source.cols})`;
  } else {
    dataSourceStatus.textContent = "Source loaded.";
  }
}

async function parseJsonResponse(res) {
  const text = await res.text();
  try {
    return { data: JSON.parse(text), text };
  } catch (err) {
    return { data: null, text };
  }
}

async function fetchDirectory(pathValue = "") {
  const res = await fetch(`/api/fs?path=${encodeURIComponent(pathValue)}`);
  const { data, text } = await parseJsonResponse(res);
  if (!res.ok || !data) {
    const detail = data?.error || text || `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return data;
}

async function fetchFile(pathValue = "", limit = null) {
  const limitParam = limit == null ? "" : `&limit=${encodeURIComponent(limit)}`;
  const res = await fetch(
    `/api/file?path=${encodeURIComponent(pathValue)}${limitParam}`
  );
  const { data, text } = await parseJsonResponse(res);
  if (!res.ok || !data) {
    const detail = data?.error || text || `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return data;
}

function screenToWorld(clientX, clientY) {
  const ctm = svg.getScreenCTM();
  if (!ctm) {
    return { x: 0, y: 0 };
  }
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const svgPoint = pt.matrixTransform(ctm.inverse());
  const x = (svgPoint.x - state.pan.x) / state.pan.scale;
  const y = (svgPoint.y - state.pan.y) / state.pan.scale;
  return { x, y };
}

function render() {
  nodeCount.textContent = `${state.nodes.length} nodes`;
  edgeCount.textContent = `${state.edges.length} edges`;
  jsonView.textContent = JSON.stringify(
    {
      nodes: state.nodes.map(serializeNode),
      edges: state.edges,
      meta: {
        generated_at: new Date().toISOString(),
        ...(state.meta || {}),
      },
    },
    null,
    2
  );

  svg.innerHTML = "";
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.setAttribute(
    "transform",
    `translate(${state.pan.x}, ${state.pan.y}) scale(${state.pan.scale})`
  );

  state.edges.forEach((edge, idx) => {
    const from = state.nodes.find((n) => n.id === edge.from);
    const to = state.nodes.find((n) => n.id === edge.to);
    if (!from || !to) return;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const startY = from.y - 30;
    const endY = to.y + 30;
    const midY = (startY + endY) / 2;
    const d = `M ${from.x} ${startY} C ${from.x} ${midY}, ${to.x} ${midY}, ${to.x} ${endY}`;
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    const isSelected = selectionState.edgeIndices.has(idx);
    path.setAttribute("stroke", isSelected ? "#c84630" : "#1c1b1a");
    path.setAttribute("stroke-width", isSelected ? "3" : "2");
    path.setAttribute("opacity", isSelected ? "0.8" : "0.4");
    g.appendChild(path);
  });

  state.nodes.forEach((node) => {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "node");
    group.setAttribute("transform", `translate(${node.x}, ${node.y})`);
    group.style.cursor = "pointer";

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", "-60");
    rect.setAttribute("y", "-30");
    rect.setAttribute("width", "120");
    rect.setAttribute("height", "60");
    rect.setAttribute("rx", "14");
    rect.setAttribute("fill", typeMeta[node.type]?.color || "#999");
    rect.setAttribute(
      "opacity",
      node.id === state.selectedId || selectionState.nodeIds.has(node.id) ? "0.9" : "0.75"
    );
    rect.setAttribute(
      "stroke",
      selectionState.nodeIds.has(node.id) ? "#1c1b1a" : "none"
    );
    rect.setAttribute(
      "stroke-width",
      selectionState.nodeIds.has(node.id) ? "2" : "0"
    );

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("y", "5");
    text.setAttribute("fill", "#fff");
    text.setAttribute("font-size", "12");
    text.setAttribute("font-family", "Space Grotesk, sans-serif");
    text.textContent = node.label;

    const tag = document.createElementNS("http://www.w3.org/2000/svg", "text");
    tag.setAttribute("text-anchor", "middle");
    tag.setAttribute("y", "24");
    tag.setAttribute("fill", "#fff");
    tag.setAttribute("font-size", "10");
    tag.textContent = node.inputKey ? `in: ${node.inputKey}` : "";

    group.appendChild(rect);
    group.appendChild(text);
    group.appendChild(tag);

    group.addEventListener("click", () => {
      if (state.draggingJustEnded) {
        state.draggingJustEnded = false;
        return;
      }
      if (state.connectMode) {
        if (!state.connectFrom) {
          state.connectFrom = node.id;
        } else {
          connectNodes(state.connectFrom, node.id);
          state.connectFrom = null;
          state.connectMode = false;
          connectButton.textContent = "Connect";
          connectButton.classList.remove("danger");
        }
        render();
        return;
      }
      if (state.clickTimer) {
        clearTimeout(state.clickTimer);
      }
      state.clickTimer = setTimeout(() => {
        selectNode(node.id);
        render();
        state.clickTimer = null;
      }, 250);
    });

    group.addEventListener("dblclick", (event) => {
      event.preventDefault();
      if (state.clickTimer) {
        clearTimeout(state.clickTimer);
        state.clickTimer = null;
      }
      if (node.type === "data") {
        openPreview(node);
      } else if (node.type === "loader") {
        openLoaderPreview(node);
      }
    });

    group.addEventListener("mousedown", (event) => {
      event.stopPropagation();
      state.draggingNodeId = node.id;
      state.draggingMoved = false;
      const world = screenToWorld(event.clientX, event.clientY);
      state.draggingOffset = { x: world.x - node.x, y: world.y - node.y };
    });

    g.appendChild(group);
  });

  svg.appendChild(g);

  if (selectionBox) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    const minX = Math.min(selectionBox.x1, selectionBox.x2);
    const minY = Math.min(selectionBox.y1, selectionBox.y2);
    const width = Math.abs(selectionBox.x2 - selectionBox.x1);
    const height = Math.abs(selectionBox.y2 - selectionBox.y1);
    rect.setAttribute("x", `${minX}`);
    rect.setAttribute("y", `${minY}`);
    rect.setAttribute("width", `${width}`);
    rect.setAttribute("height", `${height}`);
    rect.setAttribute("fill", "rgba(200,70,48,0.12)");
    rect.setAttribute("stroke", "#c84630");
    rect.setAttribute("stroke-width", "1");
    g.appendChild(rect);
  }
}

function serializeNode(node) {
  if (!node) return node;
  const base = { ...node };
  if (node.type !== "bridge") {
    delete base.bridgeMethod;
    delete base.bridgeFusionDim;
    delete base.bridgeUseSe;
    delete base.bridgeSeReduction;
    delete base.bridgeSePreNorm;
  } else if (base.bridgeMethod !== "fusion") {
    delete base.bridgeFusionDim;
    delete base.bridgeUseSe;
    delete base.bridgeSeReduction;
    delete base.bridgeSePreNorm;
  }
  if (node.type !== "filter") {
    delete base.filterType;
    delete base.regexPattern;
    delete base.columnName;
    delete base.columnOperator;
    delete base.columnValue;
  }
  if (node.type !== "transform") {
    delete base.transformType;
    delete base.roiMaskSource;
    delete base.roiScale;
    delete base.roiTargetSize;
    delete base.roiFallback;
    delete base.centerCropSize;
    delete base.jitterHFlip;
    delete base.jitterVFlip;
    delete base.jitterRotation;
    delete base.jitterColorEnabled;
    delete base.jitterColor;
    delete base.resizeSize;
  }
  const isTower = node.type === "image_tower" || node.type === "metadata_tower";
  if (isTower) {
    const towerType = node.towerType || (node.type === "metadata_tower" ? "metadata" : "image");
    base.towerType = towerType;
    if (towerType === "image") {
      delete base.mdHiddenDim;
      delete base.mdDropout;
      delete base.mdUseSe;
      delete base.mdSeReduction;
      delete base.mdSePreNorm;
      delete base.mdFreezeRatio;
    } else {
      delete base.imageBackbone;
      delete base.imageFreezeRatio;
      delete base.imageAugment;
      delete base.imageGeometryDim;
      delete base.imageUseSe;
      delete base.imageSeReduction;
      delete base.imageSePreNorm;
    }
  }
  return base;
}

function resetGraph() {
  state.nodes = [];
  state.edges = [];
  state.selectedId = null;
  state.meta = { imports: [] };
  setSelection([]);
  render();
}

function saveLocal() {
  localStorage.setItem("hypertower_v2_builder", JSON.stringify(state));
}

function loadLocal() {
  const raw = localStorage.getItem("hypertower_v2_builder");
  if (!raw) return;
  const loaded = JSON.parse(raw);
  state.nodes = loaded.nodes || [];
  state.edges = loaded.edges || [];
  state.meta = loaded.meta || { imports: [] };
  state.selectedId = null;
  setSelection([]);
  autoLayout();
  render();
}

function exportJson() {
  const blob = new Blob([jsonView.textContent], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "hypertower_v2_config.json";
  link.click();
  URL.revokeObjectURL(url);
}

function importJson(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    const data = JSON.parse(event.target.result);
    state.nodes = data.nodes || [];
    state.edges = data.edges || [];
    state.meta = data.meta || { imports: [] };
    state.selectedId = null;
    setSelection([]);
    autoLayout();
    render();
  };
  reader.readAsText(file);
}

let browseState = { path: "", entries: [], selected: null };

function openModal() {
  modalOverlay.classList.remove("hidden");
  browseState = { path: "", entries: [], selected: null };
  loadDirectory("");
}

function closeModal() {
  modalOverlay.classList.add("hidden");
}

function openPreview(node) {
  previewOverlay.classList.remove("hidden");
  previewBody.innerHTML = "";
  if (!node.source) {
    previewSubtitle.textContent = "No source selected.";
    previewBody.innerHTML = "<div class='hint'>Select a source first.</div>";
    return;
  }
  if (node.outputType === "image" && node.source.mode === "directory") {
    previewSubtitle.textContent = `Images in ${node.source.path}`;
    loadPreviewDirectory(node.source.path);
    return;
  }
  if (node.outputType === "matrix" && node.source.mode === "file") {
    previewSubtitle.textContent = `Matrix preview: ${node.source.path}`;
    loadPreviewFile(node.source.path);
    return;
  }
  if (node.outputType === "matrix" && node.source.mode === "dataframe") {
    previewSubtitle.textContent = `Matrix preview: ${node.source.name}`;
    loadPreviewPapilaDf();
    return;
  }
  previewSubtitle.textContent = "Unsupported source type.";
  previewBody.innerHTML = "<div class='hint'>Select a valid source.</div>";
}

function openLoaderPreview(node) {
  previewOverlay.classList.remove("hidden");
  previewBody.innerHTML = "";
  const loaderType = node.inputType || node.outputType || "";
  const { dataNode, filters, ambiguous } = resolveUpstreamData(node.id);
  const relevantFilters = filters
    .filter((filter) => filter.inputType === loaderType)
    .reverse();
  const useFullData = relevantFilters.length > 0 ? 0 : null;
  const summary = summarizeFilters(relevantFilters);
  const ambiguityNote = ambiguous ? " (multiple inputs: first path)" : "";
  previewSubtitle.textContent = `Loader: ${node.label}${ambiguityNote}`;
  if (summary) {
    previewSubtitle.textContent += ` | Filters: ${summary}`;
  } else if (filters.length > 0) {
    previewSubtitle.textContent += " | Filters: none applied (check input type)";
  }
  if (!dataNode) {
    previewBody.innerHTML = "<div class='hint'>No upstream data source found.</div>";
    return;
  }
  if (!dataNode.source) {
    previewBody.innerHTML = "<div class='hint'>Upstream data source has no file selected.</div>";
    return;
  }
  if (loaderType === "image") {
    if (dataNode.source.mode !== "directory") {
      previewBody.innerHTML = "<div class='hint'>Upstream source is not a directory.</div>";
      return;
    }
    loadPreviewDirectory(dataNode.source.path, relevantFilters);
    return;
  }
  if (loaderType === "matrix") {
    if (dataNode.source.mode === "file") {
      loadPreviewFile(dataNode.source.path, relevantFilters, useFullData);
      return;
    }
    if (dataNode.source.mode === "dataframe") {
      loadPreviewPapilaDf(relevantFilters, useFullData);
      return;
    }
    previewBody.innerHTML = "<div class='hint'>Upstream source is not a matrix file.</div>";
    return;
  }
  previewBody.innerHTML = "<div class='hint'>Loader input type is not set.</div>";
}

async function loadPreviewDirectory(pathValue, filters = []) {
  previewBody.innerHTML = "<div class='hint'>Loading…</div>";
  try {
    const data = await fetchDirectory(pathValue);
    const list = document.createElement("div");
    list.className = "preview-list";
    const allImages = data.entries.filter((entry) =>
      /\.(png|jpg|jpeg|tif|tiff)$/i.test(entry.name)
    );
    let files = allImages;
    const { filtered, warnings } = applyRegexFilters(allImages, filters);
    files = filtered;
    files.slice(0, 200).forEach((entry) => {
      const item = document.createElement("div");
      item.className = "preview-item";
      item.textContent = entry.name;
      list.appendChild(item);
    });
    previewBody.innerHTML = "";
    const warningEl = renderPreviewWarnings(warnings);
    if (warningEl) previewBody.appendChild(warningEl);
    const countLine = renderPreviewCount(
      "Files",
      files.length,
      allImages.length,
      files.length > 200
    );
    previewBody.appendChild(countLine);
    if (!files.length) {
      const empty = document.createElement("div");
      empty.className = "hint";
      empty.textContent = "No image files matched the filter.";
      previewBody.appendChild(empty);
      return;
    }
    previewBody.appendChild(list);
  } catch (err) {
    previewBody.innerHTML = `<div class='hint'>${err.message}</div>`;
  }
}

async function loadPreviewFile(pathValue, filters = [], limit = null) {
  previewBody.innerHTML = "<div class='hint'>Loading…</div>";
  try {
    const data = await fetchFile(pathValue, limit);
    if (!data || !data.header || !data.rows) {
      previewBody.innerHTML = "<div class='hint'>No table data.</div>";
      return;
    }
    const { filteredRows, warnings } = applyColumnFilters(data.header, data.rows, filters);
    const totalRows = Number.isFinite(data.rows_total) ? data.rows_total : data.rows.length;
    const isSampled = totalRows > data.rows.length;
    const previewRows = filteredRows.slice(0, 200);
    const table = document.createElement("table");
    table.className = "preview-table";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    data.header.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    previewRows.forEach((row) => {
      const tr = document.createElement("tr");
      row.forEach((cell) => {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    previewBody.innerHTML = "";
    const warningEl = renderPreviewWarnings(warnings);
    if (warningEl) previewBody.appendChild(warningEl);
    const countLine = renderPreviewCount(
      "Rows",
      filteredRows.length,
      totalRows,
      filteredRows.length > 200,
      isSampled ? "(preview sample)" : ""
    );
    previewBody.appendChild(countLine);
    if (!filteredRows.length) {
      const empty = document.createElement("div");
      empty.className = "hint";
      empty.textContent = "No rows matched the filter.";
      previewBody.appendChild(empty);
      return;
    }
    previewBody.appendChild(table);
  } catch (err) {
    previewBody.innerHTML = `<div class='hint'>${err.message}</div>`;
  }
}

async function loadPreviewPapilaDf(filters = [], limit = null) {
  previewBody.innerHTML = "<div class='hint'>Loading…</div>";
  try {
    const limitParam = limit == null ? "" : `?limit=${encodeURIComponent(limit)}`;
    const res = await fetch(`/api/papila/df${limitParam}`);
    const parsed = await parseJsonResponse(res);
    const data = parsed.data;
    if (!res.ok || !data) {
      throw new Error(data?.error || parsed.text || "Failed to load dataframe preview");
    }
    if (!data.header || !data.rows) {
      previewBody.innerHTML = "<div class='hint'>No table data.</div>";
      return;
    }
    const { filteredRows, warnings } = applyColumnFilters(data.header, data.rows, filters);
    const totalRows = Number.isFinite(data.rows_total) ? data.rows_total : data.rows.length;
    const isSampled = totalRows > data.rows.length;
    const previewRows = filteredRows.slice(0, 200);
    const table = document.createElement("table");
    table.className = "preview-table";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    data.header.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    previewRows.forEach((row) => {
      const tr = document.createElement("tr");
      row.forEach((cell) => {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    previewBody.innerHTML = "";
    const warningEl = renderPreviewWarnings(warnings);
    if (warningEl) previewBody.appendChild(warningEl);
    const countLine = renderPreviewCount(
      "Rows",
      filteredRows.length,
      totalRows,
      filteredRows.length > 200,
      isSampled ? "(preview sample)" : ""
    );
    previewBody.appendChild(countLine);
    if (!filteredRows.length) {
      const empty = document.createElement("div");
      empty.className = "hint";
      empty.textContent = "No rows matched the filter.";
      previewBody.appendChild(empty);
      return;
    }
    previewBody.appendChild(table);
  } catch (err) {
    previewBody.innerHTML = `<div class='hint'>${err.message}</div>`;
  }
}

async function loadDirectory(pathValue) {
  modalList.innerHTML = "Loading…";
  try {
    const data = await fetchDirectory(pathValue);
    browseState.path = data.path || "";
    browseState.entries = data.entries || [];
    browseState.selected = null;
    renderBreadcrumb();
    renderModalList();
  } catch (err) {
    modalList.innerHTML = `<div class="hint">${err.message}</div>`;
  }
}

function renderBreadcrumb() {
  breadcrumb.innerHTML = "";
  const parts = browseState.path.split("/").filter(Boolean);
  const root = document.createElement("span");
  root.textContent = "repo root";
  root.addEventListener("click", () => loadDirectory(""));
  breadcrumb.appendChild(root);
  let acc = "";
  parts.forEach((part) => {
    acc = acc ? `${acc}/${part}` : part;
    const sep = document.createElement("span");
    sep.textContent = " / ";
    breadcrumb.appendChild(sep);
    const crumb = document.createElement("span");
    crumb.textContent = part;
    crumb.addEventListener("click", () => loadDirectory(acc));
    breadcrumb.appendChild(crumb);
  });
}

function renderModalList() {
  modalList.innerHTML = "";
  browseState.entries.forEach((entry) => {
    const item = document.createElement("div");
    item.className = "modal-item";
    item.innerHTML = `<span>${entry.name}</span><span class="muted">${entry.type}</span>`;
    item.addEventListener("click", () => {
      browseState.selected = entry;
      document.querySelectorAll(".modal-item").forEach((el) => el.classList.remove("selected"));
      item.classList.add("selected");
    });
    item.addEventListener("dblclick", () => {
      if (entry.type === "dir") {
        loadDirectory(entry.relPath);
      } else {
        browseState.selected = entry;
        if (dataOutputType.value === "matrix") {
          selectFileFromModal();
        }
      }
    });
    modalList.appendChild(item);
  });
}

function selectDirectoryFromModal() {
  const node = state.nodes.find((n) => n.id === state.selectedId);
  if (!node) return;
  if (!browseState.path) {
    node.source = { mode: "directory", path: ".", count: countImages(browseState.entries) };
  } else {
    node.source = {
      mode: "directory",
      path: browseState.path,
      count: countImages(browseState.entries),
    };
  }
  updateDataSourceStatus(node);
  closeModal();
  render();
}

function selectFileFromModal() {
  const node = state.nodes.find((n) => n.id === state.selectedId);
  if (!node || !browseState.selected || browseState.selected.type !== "file") return;
  node.source = {
    mode: "file",
    path: browseState.selected.relPath,
    size: browseState.selected.size,
  };
  updateDataSourceStatus(node);
  closeModal();
  render();
}

function countImages(entries) {
  return entries.filter((entry) => entry.type === "file" && /\.(png|jpg|jpeg|tif|tiff)$/i.test(entry.name))
    .length;
}

function duplicateSelectedDataSource() {
  const node = state.nodes.find((n) => n.id === state.selectedId);
  if (!node || node.type !== "data") return;
  node.outputType = dataOutputType.value || node.outputType;
  const copy = {
    ...node,
    id: `node_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    label: `${node.label} Copy`,
    source: node.source ? { ...node.source } : null,
    x: (node.x || 140) + 40,
    y: (node.y || 120) + 120,
  };
  state.nodes.push(copy);
  selectNode(copy.id);
  render();
}

function loadCustomPresets() {
  try {
    const raw = localStorage.getItem(PRESET_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (err) {
    return {};
  }
}

function saveCustomPresets(presets) {
  localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(presets, null, 2));
}

function buildBuiltinPresets() {
  return {
    roi_center_jitter: {
      label: "ROI → CenterCrop → Jitter",
      nodes: [
        {
          id: "roi",
          type: "transform",
          label: "ROI Crop",
          transformType: "roi_crop",
          roiMaskSource: "gt",
          roiScale: 2.5,
          roiTargetSize: 224,
          roiFallback: true,
          x: 0,
          y: 0,
        },
        {
          id: "center",
          type: "transform",
          label: "Center Crop",
          transformType: "center_crop",
          centerCropSize: 224,
          x: 0,
          y: -120,
        },
        {
          id: "jitter",
          type: "transform",
          label: "Jitter Bundle",
          transformType: "jitter_bundle",
          jitterHFlip: true,
          jitterVFlip: true,
          jitterRotation: 15,
          jitterColorEnabled: true,
          jitterColor: "0.1,0.1,0.1,0.05",
          x: 0,
          y: -240,
        },
      ],
      edges: [
        { from: "roi", to: "center" },
        { from: "center", to: "jitter" },
      ],
    },
  };
}

function refreshPresetSelect() {
  const custom = loadCustomPresets();
  const builtins = buildBuiltinPresets();
  presetSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select…";
  presetSelect.appendChild(placeholder);
  Object.entries(builtins).forEach(([key, preset]) => {
    const option = document.createElement("option");
    option.value = `builtin:${key}`;
    option.textContent = preset.label || key;
    presetSelect.appendChild(option);
  });
  Object.entries(custom).forEach(([key, preset]) => {
    const option = document.createElement("option");
    option.value = `custom:${key}`;
    option.textContent = `Custom: ${preset.label || key}`;
    presetSelect.appendChild(option);
  });
}

function buildPresetFromSelection(name) {
  const selectedIds = Array.from(selectionState.nodeIds);
  if (!selectedIds.length) return null;
  const nodes = state.nodes.filter((n) => selectionState.nodeIds.has(n.id));
  const edges = state.edges.filter(
    (e) => selectionState.nodeIds.has(e.from) && selectionState.nodeIds.has(e.to)
  );
  const xs = nodes.map((n) => n.x ?? 0);
  const ys = nodes.map((n) => n.y ?? 0);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const normalizedNodes = nodes.map((node) => {
    const { id, ...rest } = node;
    return {
      id,
      ...rest,
      x: (node.x ?? 0) - minX,
      y: (node.y ?? 0) - minY,
    };
  });
  return {
    label: name,
    nodes: normalizedNodes,
    edges: edges.map((edge) => ({ ...edge })),
  };
}

function applyPreset(preset) {
  if (!preset || !preset.nodes || !preset.nodes.length) return;
  const svgRect = svg.getBoundingClientRect();
  const center = screenToWorld(
    svgRect.left + svgRect.width / 2,
    svgRect.top + svgRect.height / 2
  );
  const xs = preset.nodes.map((n) => n.x ?? 0);
  const ys = preset.nodes.map((n) => n.y ?? 0);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);
  const width = maxX - minX;
  const height = maxY - minY;
  const offsetX = center.x - width / 2 - minX;
  const offsetY = center.y - height / 2 - minY;

  const idMap = new Map();
  const newNodes = preset.nodes.map((node, idx) => {
    const base = createNodeTemplate(node.type || "transform");
    const { id, x, y, ...rest } = node;
    Object.assign(base, rest);
    base.id = newNodeId();
    base.label = node.label || base.label;
    base.x = (x ?? idx * 80) + offsetX;
    base.y = (y ?? idx * -80) + offsetY;
    idMap.set(id ?? `idx_${idx}`, base.id);
    return base;
  });
  const newEdges = (preset.edges || [])
    .map((edge) => ({
      from: idMap.get(edge.from),
      to: idMap.get(edge.to),
    }))
    .filter((edge) => edge.from && edge.to);
  state.nodes.push(...newNodes);
  state.edges.push(...newEdges);
  setSelection(newNodes.map((n) => n.id));
  if (newNodes.length === 1) {
    selectNode(newNodes[0].id);
  } else {
    state.selectedId = null;
    clearSelectionUI();
  }
  render();
}

function setupPanZoom() {
  let dragging = false;
  let last = { x: 0, y: 0 };
  svg.addEventListener("mousedown", (e) => {
    if (e.target.closest(".node")) return;
    if (e.shiftKey) {
      const world = screenToWorld(e.clientX, e.clientY);
      selectionBox = { x1: world.x, y1: world.y, x2: world.x, y2: world.y };
      render();
      return;
    }
    dragging = true;
    last = { x: e.clientX, y: e.clientY };
  });
  window.addEventListener("mousemove", (e) => {
    if (selectionBox) {
      const world = screenToWorld(e.clientX, e.clientY);
      selectionBox.x2 = world.x;
      selectionBox.y2 = world.y;
      render();
      return;
    }
    if (state.draggingNodeId) {
      const node = state.nodes.find((n) => n.id === state.draggingNodeId);
      if (!node || !state.draggingOffset) return;
      const world = screenToWorld(e.clientX, e.clientY);
      node.x = world.x - state.draggingOffset.x;
      node.y = world.y - state.draggingOffset.y;
      if (!state.draggingMoved) {
        state.draggingMoved = true;
      }
      render();
      return;
    }
    if (!dragging) return;
    const dx = e.clientX - last.x;
    const dy = e.clientY - last.y;
    state.pan.x += dx;
    state.pan.y += dy;
    last = { x: e.clientX, y: e.clientY };
    render();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    if (state.draggingNodeId) {
      state.draggingJustEnded = state.draggingMoved;
      state.draggingNodeId = null;
      state.draggingOffset = null;
      state.draggingMoved = false;
    }
    if (selectionBox) {
      const minX = Math.min(selectionBox.x1, selectionBox.x2);
      const maxX = Math.max(selectionBox.x1, selectionBox.x2);
      const minY = Math.min(selectionBox.y1, selectionBox.y2);
      const maxY = Math.max(selectionBox.y1, selectionBox.y2);
      const selected = state.nodes
        .filter((node) => node.x >= minX && node.x <= maxX && node.y >= minY && node.y <= maxY)
        .map((node) => node.id);
      setSelection(selected);
      if (selected.length === 1) {
        selectNode(selected[0]);
      } else {
        state.selectedId = null;
        clearSelectionUI();
      }
      selectionBox = null;
      render();
    }
  });
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1.05 : 0.95;
    state.pan.scale = Math.max(0.5, Math.min(2.0, state.pan.scale * delta));
    render();
  });
}

document.querySelectorAll("[data-add]").forEach((btn) => {
  btn.addEventListener("click", () => addNode(btn.dataset.add));
});

document.getElementById("apply-node").addEventListener("click", applyNodeEdits);
document.getElementById("delete-node").addEventListener("click", deleteSelected);
document.getElementById("connect-mode").addEventListener("click", toggleConnectMode);
document.getElementById("disconnect-selected").addEventListener("click", removeSelectedEdges);
document.getElementById("auto-layout").addEventListener("click", () => {
  autoLayout();
  render();
});
document.getElementById("reset-graph").addEventListener("click", resetGraph);
document.getElementById("save-local").addEventListener("click", saveLocal);
document.getElementById("load-local").addEventListener("click", loadLocal);
document.getElementById("export-json").addEventListener("click", exportJson);
document.getElementById("import-json").addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    importJson(e.target.files[0]);
  }
});
importClassBtn.addEventListener("click", importClassDefinition);
presetApplyBtn.addEventListener("click", () => {
  const value = presetSelect.value;
  if (!value) return;
  const [kind, key] = value.split(":");
  if (kind === "builtin") {
    const preset = buildBuiltinPresets()[key];
    applyPreset(preset);
    return;
  }
  if (kind === "custom") {
    const custom = loadCustomPresets();
    const preset = custom[key];
    applyPreset(preset);
  }
});
presetSaveBtn.addEventListener("click", () => {
  if (!selectionState.nodeIds.size) {
    alert("Select nodes to save as a preset (Shift + drag on the graph).");
    return;
  }
  const name = window.prompt("Preset name:");
  if (!name) return;
  const preset = buildPresetFromSelection(name);
  if (!preset) return;
  const custom = loadCustomPresets();
  custom[name] = preset;
  saveCustomPresets(custom);
  refreshPresetSelect();
  presetSelect.value = `custom:${name}`;
});

loaderInputType.addEventListener("change", () => {
  loaderOutputType.value = loaderInputType.value;
  applyNodeEdits();
});

loaderInputIndex.addEventListener("change", applyNodeEdits);

filterInputType.addEventListener("change", () => {
  filterOutputType.value = filterInputType.value;
  applyNodeEdits();
});

filterType.addEventListener("change", () => {
  updateFilterFieldVisibility(filterType.value);
  applyNodeEdits();
});

filterRegexPattern.addEventListener("change", applyNodeEdits);
filterColumnName.addEventListener("change", applyNodeEdits);
filterColumnOperator.addEventListener("change", applyNodeEdits);
filterColumnValue.addEventListener("change", applyNodeEdits);

transformType.addEventListener("change", () => {
  updateTransformFieldVisibility(transformType.value);
  applyNodeEdits();
});
transformRoiMaskSource.addEventListener("change", applyNodeEdits);
transformRoiScale.addEventListener("change", applyNodeEdits);
transformRoiTarget.addEventListener("change", applyNodeEdits);
transformRoiFallback.addEventListener("change", applyNodeEdits);
transformCenterSize.addEventListener("change", applyNodeEdits);
transformJitterHFlip.addEventListener("change", applyNodeEdits);
transformJitterVFlip.addEventListener("change", applyNodeEdits);
transformJitterRotation.addEventListener("change", applyNodeEdits);
transformJitterColorEnabled.addEventListener("change", applyNodeEdits);
transformJitterColor.addEventListener("change", applyNodeEdits);
transformResizeSize.addEventListener("change", applyNodeEdits);

towerType.addEventListener("change", () => {
  updateTowerFieldVisibility(towerType.value);
  applyNodeEdits();
});
towerBackbone.addEventListener("change", applyNodeEdits);
towerFreezeRatio.addEventListener("change", applyNodeEdits);
towerAugment.addEventListener("change", applyNodeEdits);
towerGeometryDim.addEventListener("change", applyNodeEdits);
towerUseSe.addEventListener("change", applyNodeEdits);
towerSeReduction.addEventListener("change", applyNodeEdits);
towerSePreNorm.addEventListener("change", applyNodeEdits);
towerMdHidden.addEventListener("change", applyNodeEdits);
towerMdDropout.addEventListener("change", applyNodeEdits);
towerMdUseSe.addEventListener("change", applyNodeEdits);
towerMdSeReduction.addEventListener("change", applyNodeEdits);
towerMdSePreNorm.addEventListener("change", applyNodeEdits);
towerMdFreezeRatio.addEventListener("change", applyNodeEdits);

bridgeMethod.addEventListener("change", () => {
  updateBridgeFieldVisibility(bridgeMethod.value);
  applyNodeEdits();
});
bridgeFusionDim.addEventListener("change", applyNodeEdits);
bridgeUseSe.addEventListener("change", applyNodeEdits);
bridgeSeReduction.addEventListener("change", applyNodeEdits);
bridgeSePreNorm.addEventListener("change", applyNodeEdits);

dataOutputType.addEventListener("change", () => {
  const node = state.nodes.find((n) => n.id === state.selectedId);
  if (node) {
    node.outputType = dataOutputType.value;
    node.source = null;
  }
  updateDataBrowseVisibility(dataOutputType.value);
  updateDataSourceStatus(node);
  render();
});

dataBrowseButton.addEventListener("click", () => {
  if (dataOutputType.value) {
    modalMode.textContent =
      dataOutputType.value === "image"
        ? "Select an image directory."
        : "Select a CSV/TSV/TXT file.";
    modalSelectDir.disabled = dataOutputType.value !== "image";
    modalSelectFile.disabled = dataOutputType.value !== "matrix";
    openModal();
  }
});

duplicateDataButton.addEventListener("click", duplicateSelectedDataSource);

modalClose.addEventListener("click", closeModal);
modalSelectDir.addEventListener("click", selectDirectoryFromModal);
modalSelectFile.addEventListener("click", selectFileFromModal);

modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) {
    closeModal();
  }
});

previewClose.addEventListener("click", () => {
  previewOverlay.classList.add("hidden");
});

previewOverlay.addEventListener("click", (e) => {
  if (e.target === previewOverlay) {
    previewOverlay.classList.add("hidden");
  }
});

toggleJsonButton.addEventListener("click", () => {
  jsonView.classList.toggle("collapsed");
  toggleJsonButton.textContent = jsonView.classList.contains("collapsed")
    ? "Expand"
    : "Collapse";
});

setupPanZoom();
resetGraph();
refreshPresetSelect();
