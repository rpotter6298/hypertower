# classes/backbones.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch import nn
from torchvision import models

@dataclass(frozen=True)
class BackboneSpec:
    ctor: Callable                  # torchvision constructor
    weights_default: object         # torchvision Weights enum DEFAULT member
    strip: Callable[[nn.Module], tuple]   # fn(model)->(out_dim, model_no_head)
    blocks: Callable[[nn.Module], List[nn.Module]]  # fn(model)->ordered blocks for freezing

REFUGELIKE_BACKBONE_PATH = Path("models/v2/refuge/refugelike_backbone.pt")
REFUGE_DENSENET_PATH = Path("models/refuge/classifier/refuge_densenet_backbone.pt")
REFUGE_EFFICIENT_B0_PATH = Path("models/refuge/classifier/refuge_efficient_b0_backbone.pt")
REFUGE_EFFICIENT_B7_PATH = Path("models/refuge/classifier/refuge_efficient_b7_backbone.pt")

# --- strip fns ---
def _strip_efficientnet_b0(m: models.EfficientNet):
    from torch import nn as _nn
    out_dim = m.classifier[1].in_features
    m.classifier = _nn.Identity()
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
    out_dim = m.classifier[0].in_features   # 25088 for VGG16 at 224×224
    m.classifier = nn.Identity()
    return out_dim, m

def _strip_mobilenet_v2(m: models.MobileNetV2):
    out_dim = m.classifier[1].in_features
    m.classifier = nn.Identity()
    return out_dim, m

def _strip_inception_v3(m: models.Inception3):
    out_dim = m.fc.in_features
    m.fc = nn.Identity()
    if hasattr(m, "AuxLogits"):
        m.aux_logits = False
    return out_dim, m

# --- block splitters for ratio-based freezing ---
def _blocks_efficientnet_b0(m: models.EfficientNet):
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
    if cur: stages.append(nn.Sequential(*cur))
    return stages

def _blocks_mobilenet_v2(m: models.MobileNetV2):
    return list(m.features)

def _blocks_inception_v3(m: models.Inception3):
    blocks = []
    for name, child in m.named_children():
        if name in ("fc", "AuxLogits"):
            continue
        blocks.append(child)
    return blocks

# --- registry (covers paper models available in torchvision) ---
BACKBONES: Dict[str, BackboneSpec] = {
    "efficientnet_b0": BackboneSpec(
        ctor=models.efficientnet_b0,
        weights_default=models.EfficientNet_B0_Weights.DEFAULT,
        strip=_strip_efficientnet_b0,
        blocks=_blocks_efficientnet_b0,
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
        strip=_strip_efficientnet_b0,
        blocks=_blocks_efficientnet_b0,
    ),
    "refuge_efficient_b7": BackboneSpec(
        ctor=models.efficientnet_b7,
        weights_default=None,
        strip=_strip_efficientnet_b0,
        blocks=_blocks_efficientnet_b0,
    ),
    # Xception isn’t in torchvision 
}

def list_names() -> List[str]:
    return list(BACKBONES.keys())
    

def load_backbone_weights(key: str, model: nn.Module) -> None:
    if key == "refugelike":
        path = REFUGELIKE_BACKBONE_PATH
    elif key == "refuge_densenet":
        path = REFUGE_DENSENET_PATH
    elif key == "refuge_efficient_b0":
        path = REFUGE_EFFICIENT_B0_PATH
    elif key == "refuge_efficient_b7":
        path = REFUGE_EFFICIENT_B7_PATH
    else:
        return

    if not path.exists():
        raise FileNotFoundError(
            "Custom REFUGE backbone not found at "
            f"{path}. Export it via refuge_build.py --export-backbone first."
        )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
