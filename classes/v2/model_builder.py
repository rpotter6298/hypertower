from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import nn

from classes.v2.bridges import Bridge, VoteBridge
from classes.v2.towers import ImageTower, MDTower

from .config_builder import ConfigAssembly
from .transforms import build_transform_chain


@dataclass
class V2ModelBundle:
    image_tower: Optional[ImageTower]
    metadata_tower: Optional[MDTower]
    bridge: Optional[nn.Module]
    classifier: Optional[nn.Module]
    image_transform: Optional[Callable]
    matrix_transform: Optional[Callable]


def build_model_bundle(
    assembly: ConfigAssembly,
    clinical: Any,
    *,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> V2ModelBundle:
    """
    Build torch modules and input transforms from a V2 config assembly.
    """
    image_tower_spec = _pick_tower(assembly, "image")
    md_tower_spec = _pick_tower(assembly, "metadata")
    bridge_spec = _pick_bridge(assembly)
    image_loader = _pick_loader(assembly, input_type="image")

    clinical_core = getattr(clinical, "clinical", clinical)
    num_classes = _infer_num_classes(clinical)

    img_tower = None
    if image_tower_spec is not None:
        img_tower = ImageTower(
            backbone=image_tower_spec.params.get("backbone", "efficientnet_b0"),
            freeze_ratio=float(image_tower_spec.params.get("freeze_ratio", 0.0) or 0.0),
            use_se=bool(image_tower_spec.params.get("use_se", False)),
            se_reduction=int(image_tower_spec.params.get("se_reduction", 16) or 16),
            se_pre_norm=bool(image_tower_spec.params.get("se_pre_norm", True)),
            augment=bool(image_tower_spec.params.get("augment", True)),
            geometry_dim=int(image_tower_spec.params.get("geometry_dim", 0) or 0),
        )
        if device is not None:
            img_tower = img_tower.to(device)

    md_tower = None
    if md_tower_spec is not None:
        md_tower = MDTower(
            clinical_core,
            hidden_dim=int(md_tower_spec.params.get("hidden_dim", 128) or 128),
            dropout=float(md_tower_spec.params.get("dropout", 0.1) or 0.1),
            use_se=bool(md_tower_spec.params.get("use_se", False)),
            se_reduction=int(md_tower_spec.params.get("se_reduction", 16) or 16),
            se_pre_norm=bool(md_tower_spec.params.get("se_pre_norm", True)),
        )
        if device is not None:
            md_tower = md_tower.to(device)

    bridge = None
    if bridge_spec is not None and img_tower is not None and md_tower is not None:
        if bridge_spec.method == "consensus":
            bridge = VoteBridge(num_classes=num_classes)
        else:
            bridge = Bridge(
                img_dim=img_tower.out_dim,
                meta_dim=md_tower.out_dim,
                num_classes=num_classes,
                fusion_dim=int(bridge_spec.params.get("fusion_dim", 256) or 256),
                mode="fused",
                use_se=bool(bridge_spec.params.get("use_se", True)),
                se_reduction=int(bridge_spec.params.get("se_reduction", 16) or 16),
                se_pre_norm=bool(bridge_spec.params.get("se_pre_norm", True)),
            )
        if device is not None:
            bridge = bridge.to(device)

    classifier = None
    if assembly.classifiers:
        classifier = nn.Identity()
        if device is not None:
            classifier = classifier.to(device)

    image_transform = None
    if image_loader is not None and image_tower_spec is not None:
        image_transform = build_transform_chain(
            image_loader.transforms,
            backbone_name=image_tower_spec.params.get("backbone", "efficientnet_b0"),
            augment=bool(image_tower_spec.params.get("augment", True)),
            strict=strict,
        )

    return V2ModelBundle(
        image_tower=img_tower,
        metadata_tower=md_tower,
        bridge=bridge,
        classifier=classifier,
        image_transform=image_transform,
        matrix_transform=None,
    )


def _pick_tower(assembly: ConfigAssembly, tower_type: str):
    matches = [tower for tower in assembly.towers.values() if tower.tower_type == tower_type]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple {tower_type} towers found; only one is supported for now.")
    return matches[0]


def _pick_bridge(assembly: ConfigAssembly):
    if not assembly.bridges:
        return None
    if len(assembly.bridges) > 1:
        raise ValueError("Multiple bridges found; only one is supported for now.")
    return next(iter(assembly.bridges.values()))


def _pick_loader(assembly: ConfigAssembly, input_type: str):
    matches = [loader for loader in assembly.loaders.values() if loader.input_type == input_type]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple loaders with input_type={input_type!r} found.")
    return matches[0]


def _infer_num_classes(clinical: Any) -> int:
    df = getattr(clinical, "df", None)
    label_col = getattr(clinical, "label_col", None)
    if df is None and hasattr(clinical, "clinical"):
        df = clinical.clinical.df
        label_col = clinical.clinical.label_col
    if df is None or label_col is None or label_col not in df.columns:
        return 2
    return int(df[label_col].dropna().nunique())
