"""backbones — backbone registry and builder for v4 image towers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch import nn
from torchvision import models, transforms

_REPO_ROOT = Path(__file__).resolve().parents[3]

REFUGELIKE_BACKBONE_PATH    = _REPO_ROOT / "models/v2/refuge/refugelike_backbone.pt"
REFUGE_DENSENET_PATH        = _REPO_ROOT / "models/refuge/classifier/refuge_densenet_backbone.pt"
REFUGE_EFFICIENT_B0_PATH    = _REPO_ROOT / "models/refuge/classifier/refuge_efficient_b0_backbone.pt"
REFUGE_EFFICIENT_B7_PATH    = _REPO_ROOT / "models/refuge/classifier/refuge_efficient_b7_backbone.pt"


@dataclass(frozen=True)
class BackboneSpec:
    ctor:            Callable
    weights_default: object
    strip:           Callable[[nn.Module], tuple]
    blocks:          Callable[[nn.Module], List[nn.Module]]


# ---------------------------------------------------------------------------
# Strip helpers
# ---------------------------------------------------------------------------

def _strip_efficientnet(m: models.EfficientNet):
    out_dim = m.classifier[1].in_features
    m.classifier = nn.Identity()
    return out_dim, m

def _strip_resnet(m: models.ResNet):
    out_dim = m.fc.in_features
    m.fc = nn.Identity()
    return out_dim, m

def _strip_densenet(m: models.DenseNet):
    out_dim = m.classifier.in_features
    m.classifier = nn.Identity()
    return out_dim, m

def _strip_vgg(m: models.VGG):
    out_dim = m.classifier[0].in_features
    m.classifier = nn.Identity()
    return out_dim, m

def _strip_mobilenet_v2(m: models.MobileNetV2):
    out_dim = m.classifier[1].in_features
    m.classifier = nn.Identity()
    return out_dim, m

def _strip_inception_v3(m: models.Inception3):
    out_dim = m.fc.in_features
    m.fc = nn.Identity()
    m.aux_logits = False
    m.AuxLogits = None
    return out_dim, m


# ---------------------------------------------------------------------------
# Block splitters for ratio-based freezing
# ---------------------------------------------------------------------------

def _blocks_efficientnet(m: models.EfficientNet):
    return list(m.features)

def _blocks_resnet(m: models.ResNet):
    stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
    return [stem, m.layer1, m.layer2, m.layer3, m.layer4]

def _blocks_densenet(m: models.DenseNet):
    f = m.features
    stem = nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0)
    return [stem, f.denseblock1, f.transition1, f.denseblock2, f.transition2,
            f.denseblock3, f.transition3, f.denseblock4, f.norm5]

def _blocks_vgg(m: models.VGG):
    stages, cur = [], []
    for mod in m.features:
        cur.append(mod)
        if isinstance(mod, nn.MaxPool2d):
            stages.append(nn.Sequential(*cur)); cur = []
    if cur:
        stages.append(nn.Sequential(*cur))
    return stages

def _blocks_mobilenet_v2(m: models.MobileNetV2):
    return list(m.features)

def _blocks_inception_v3(m: models.Inception3):
    return [child for name, child in m.named_children()
            if name not in ("fc", "AuxLogits")]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BACKBONES: Dict[str, BackboneSpec] = {
    "efficientnet_b0": BackboneSpec(
        ctor=models.efficientnet_b0,
        weights_default=models.EfficientNet_B0_Weights.DEFAULT,
        strip=_strip_efficientnet,
        blocks=_blocks_efficientnet,
    ),
    "resnet50": BackboneSpec(
        ctor=models.resnet50,
        weights_default=models.ResNet50_Weights.DEFAULT,
        strip=_strip_resnet,
        blocks=_blocks_resnet,
    ),
    "densenet121": BackboneSpec(
        ctor=models.densenet121,
        weights_default=models.DenseNet121_Weights.DEFAULT,
        strip=_strip_densenet,
        blocks=_blocks_densenet,
    ),
    "vgg16": BackboneSpec(
        ctor=models.vgg16,
        weights_default=models.VGG16_Weights.DEFAULT,
        strip=_strip_vgg,
        blocks=_blocks_vgg,
    ),
    "mobilenet_v2": BackboneSpec(
        ctor=models.mobilenet_v2,
        weights_default=models.MobileNet_V2_Weights.DEFAULT,
        strip=_strip_mobilenet_v2,
        blocks=_blocks_mobilenet_v2,
    ),
    "inception_v3": BackboneSpec(
        ctor=models.inception_v3,
        weights_default=models.Inception_V3_Weights.DEFAULT,
        strip=_strip_inception_v3,
        blocks=_blocks_inception_v3,
    ),
    "refugelike": BackboneSpec(
        ctor=models.resnet50,
        weights_default=None,
        strip=_strip_resnet,
        blocks=_blocks_resnet,
    ),
    "refuge_densenet": BackboneSpec(
        ctor=models.densenet121,
        weights_default=None,
        strip=_strip_densenet,
        blocks=_blocks_densenet,
    ),
    "refuge_efficient_b0": BackboneSpec(
        ctor=models.efficientnet_b0,
        weights_default=None,
        strip=_strip_efficientnet,
        blocks=_blocks_efficientnet,
    ),
    "refuge_efficient_b7": BackboneSpec(
        ctor=models.efficientnet_b7,
        weights_default=None,
        strip=_strip_efficientnet,
        blocks=_blocks_efficientnet,
    ),
}


def list_names() -> List[str]:
    return list(BACKBONES.keys())


def load_backbone_weights(key: str, model: nn.Module) -> None:
    paths = {
        "refugelike":         REFUGELIKE_BACKBONE_PATH,
        "refuge_densenet":    REFUGE_DENSENET_PATH,
        "refuge_efficient_b0": REFUGE_EFFICIENT_B0_PATH,
        "refuge_efficient_b7": REFUGE_EFFICIENT_B7_PATH,
    }
    path = paths.get(key)
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(
            f"Custom backbone weights not found at {path}. "
            "Export them via refuge_build.py --export-backbone first."
        )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)


def build_backbone(name: str, freeze_ratio: float = 0.0) -> tuple[nn.Module, int, list]:
    """Instantiate a backbone, strip its classifier head, apply freeze ratio.

    Returns (model, out_dim, blocks) where blocks is the ordered list of
    freezable units — callers use it to dynamically adjust freeze_ratio later.
    """
    key = (name or "").lower()
    if key not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list_names()}")

    spec = BACKBONES[key]
    if spec.weights_default is not None:
        m = spec.ctor(weights=spec.weights_default)
    else:
        m = spec.ctor(weights=None)
    out_dim, m = spec.strip(m)
    load_backbone_weights(key, m)

    blocks  = spec.blocks(m)
    fr      = max(0.0, min(1.0, float(freeze_ratio)))
    n_freeze = int(math.floor(len(blocks) * fr))
    for b in blocks[:n_freeze]:
        for p in b.parameters():
            p.requires_grad = False

    return m, out_dim, blocks
