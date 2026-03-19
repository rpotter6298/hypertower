from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json

from v3.classes.papila_data import PapilaData


@dataclass
class ImportSpec:
    id: str
    class_name: str
    params: Dict[str, Any]


@dataclass
class DataSourceSpec:
    node_id: str
    label: str
    output_type: str
    source: Optional[Dict[str, Any]]
    source_ref: Optional[Dict[str, Any]]


@dataclass
class TransformSpec:
    node_id: str
    label: str
    transform_type: str
    params: Dict[str, Any]


@dataclass
class LoaderSpec:
    node_id: str
    label: str
    input_type: str
    input_index: str
    input_key: str
    output_key: str
    transforms: List[TransformSpec]
    data_source: Optional[DataSourceSpec]


@dataclass
class TowerSpec:
    node_id: str
    label: str
    tower_type: str
    params: Dict[str, Any]


@dataclass
class BridgeSpec:
    node_id: str
    label: str
    method: str
    params: Dict[str, Any]


@dataclass
class ClassifierSpec:
    node_id: str
    label: str


@dataclass
class ConfigAssembly:
    raw: Dict[str, Any]
    imports: Dict[str, ImportSpec]
    data_sources: Dict[str, DataSourceSpec]
    transforms: Dict[str, TransformSpec]
    loaders: Dict[str, LoaderSpec]
    towers: Dict[str, TowerSpec]
    bridges: Dict[str, BridgeSpec]
    classifiers: Dict[str, ClassifierSpec]


def load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object.")
    return payload


def assemble_config(path: Path) -> ConfigAssembly:
    config = load_config(path)
    meta = config.get("meta", {})
    imports = _build_imports(meta.get("imports", []))
    nodes = {node["id"]: node for node in config.get("nodes", [])}
    edges = config.get("edges", [])

    data_sources: Dict[str, DataSourceSpec] = {}
    transforms: Dict[str, TransformSpec] = {}
    loaders: Dict[str, LoaderSpec] = {}
    towers: Dict[str, TowerSpec] = {}
    bridges: Dict[str, BridgeSpec] = {}
    classifiers: Dict[str, ClassifierSpec] = {}

    for node in nodes.values():
        ntype = node.get("type")
        if ntype == "data":
            data_sources[node["id"]] = DataSourceSpec(
                node_id=node["id"],
                label=node.get("label", ""),
                output_type=node.get("outputType", ""),
                source=node.get("source"),
                source_ref=node.get("sourceRef"),
            )
        elif ntype == "transform":
            transforms[node["id"]] = TransformSpec(
                node_id=node["id"],
                label=node.get("label", ""),
                transform_type=node.get("transformType", ""),
                params=_extract_transform_params(node),
            )
        elif ntype == "loader":
            loaders[node["id"]] = LoaderSpec(
                node_id=node["id"],
                label=node.get("label", ""),
                input_type=node.get("inputType", ""),
                input_index=node.get("inputIndex", ""),
                input_key=node.get("inputKey", ""),
                output_key=node.get("outputKey", ""),
                transforms=[],
                data_source=None,
            )
        elif ntype in ("image_tower", "metadata_tower"):
            towers[node["id"]] = TowerSpec(
                node_id=node["id"],
                label=node.get("label", ""),
                tower_type=node.get("towerType", "image" if ntype == "image_tower" else "clinical data"),
                params=_extract_tower_params(node),
            )
        elif ntype == "bridge":
            bridges[node["id"]] = BridgeSpec(
                node_id=node["id"],
                label=node.get("label", ""),
                method=node.get("bridgeMethod", "fusion"),
                params=_extract_bridge_params(node),
            )
        elif ntype == "classifier":
            classifiers[node["id"]] = ClassifierSpec(
                node_id=node["id"],
                label=node.get("label", ""),
            )

    # attach transforms + data sources to loaders by walking upstream
    for loader_id, loader in loaders.items():
        chain = _upstream_chain(loader_id, nodes, edges)
        for node_id in reversed(chain):
            if node_id in transforms:
                loader.transforms.append(transforms[node_id])
            if node_id in data_sources:
                loader.data_source = data_sources[node_id]

    return ConfigAssembly(
        raw=config,
        imports=imports,
        data_sources=data_sources,
        transforms=transforms,
        loaders=loaders,
        towers=towers,
        bridges=bridges,
        classifiers=classifiers,
    )


def resolve_imports(assembly: ConfigAssembly) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for import_id, spec in assembly.imports.items():
        if spec.class_name == "PapilaData":
            params = spec.params
            resolved[import_id] = PapilaData.from_dirs(
                image_dir=params.get("image_dir", "Papila/FundusImages"),
                clinical_dir=params.get("clinical_dir", "Papila/ClinicalData"),
                label_col=params.get("label_col", "Diagnosis"),
                cat_cols=params.get("cat_cols", ["Gender", "Phakic/Pseudophakic"]),
            )
        else:
            raise ValueError(f"Unsupported import class {spec.class_name!r}")
    return resolved


def _build_imports(entries: Iterable[Dict[str, Any]]) -> Dict[str, ImportSpec]:
    specs: Dict[str, ImportSpec] = {}
    for entry in entries or []:
        import_id = entry.get("id")
        if not import_id:
            continue
        specs[import_id] = ImportSpec(
            id=import_id,
            class_name=entry.get("className", ""),
            params=entry.get("params", {}) or {},
        )
    return specs


def _extract_transform_params(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "transformType": node.get("transformType"),
        "roiMaskSource": node.get("roiMaskSource"),
        "roiScale": node.get("roiScale"),
        "roiTargetSize": node.get("roiTargetSize"),
        "roiFallback": node.get("roiFallback"),
        "centerCropSize": node.get("centerCropSize"),
        "jitterHFlip": node.get("jitterHFlip"),
        "jitterVFlip": node.get("jitterVFlip"),
        "jitterRotation": node.get("jitterRotation"),
        "jitterColorEnabled": node.get("jitterColorEnabled"),
        "jitterColor": node.get("jitterColor"),
        "resizeSize": node.get("resizeSize"),
    }


def _extract_tower_params(node: Dict[str, Any]) -> Dict[str, Any]:
    if node.get("towerType") == "clinical data":
        return {
            "hidden_dim": node.get("mdHiddenDim"),
            "dropout": node.get("mdDropout"),
            "use_se": node.get("mdUseSe"),
            "se_reduction": node.get("mdSeReduction"),
            "se_pre_norm": node.get("mdSePreNorm"),
            "freeze_ratio": node.get("mdFreezeRatio"),
        }
    return {
        "backbone": node.get("imageBackbone"),
        "freeze_ratio": node.get("imageFreezeRatio"),
        "augment": node.get("imageAugment"),
        "geometry_dim": node.get("imageGeometryDim"),
        "use_se": node.get("imageUseSe"),
        "se_reduction": node.get("imageSeReduction"),
        "se_pre_norm": node.get("imageSePreNorm"),
    }


def _extract_bridge_params(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fusion_dim": node.get("bridgeFusionDim"),
        "use_se": node.get("bridgeUseSe"),
        "se_reduction": node.get("bridgeSeReduction"),
        "se_pre_norm": node.get("bridgeSePreNorm"),
    }


def _edge_from(edge: Dict[str, Any]) -> Optional[str]:
    return edge.get("from") or edge.get("source")


def _edge_to(edge: Dict[str, Any]) -> Optional[str]:
    return edge.get("to") or edge.get("target")


def _upstream_chain(start_id: str, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
    chain: List[str] = []
    visited = set()
    current = start_id
    while True:
        if current in visited:
            break
        visited.add(current)
        incoming = [edge for edge in edges if _edge_to(edge) == current]
        if not incoming:
            break
        # prefer first incoming edge for now
        current = _edge_from(incoming[0])
        if not current:
            break
        chain.append(current)
        node = nodes.get(current)
        if node and node.get("type") == "data":
            break
    return chain
