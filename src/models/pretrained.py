"""ImageNet-pretrained ResNet18 / MobileNetV2 and the stage freeze/unfreeze helpers.

Stage configs name parameters by module-name prefix: ``all`` matches every
parameter, ``layer4`` matches ``layer4.*`` and ``features.14:`` matches
``features[14]`` to the end.
"""
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    """ResNet18 with ``fc`` replaced by Dropout -> Linear(512, num_classes)."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def build_mobilenet_v2(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    """MobileNetV2 with ``classifier`` replaced by Dropout -> Linear(1280, num_classes)."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
    return model


def matches(name: str, prefix: str) -> bool:
    """True if parameter ``name`` lies under the module prefix ``prefix``."""
    if prefix == "all":
        return True
    if prefix.endswith(":"):
        base, start = prefix[:-1].rsplit(".", 1)
        if not name.startswith(base + "."):
            return False
        idx = name[len(base) + 1:].split(".", 1)[0]
        return idx.isdigit() and int(idx) >= int(start)
    return name == prefix or name.startswith(prefix + ".")


def set_trainable(model: nn.Module, prefixes: list[str]) -> int:
    """Freeze every parameter not under ``prefixes``; returns the number of trainable params."""
    used, n_trainable = set(), 0
    for name, p in model.named_parameters():
        hits = [q for q in prefixes if matches(name, q)]
        p.requires_grad = bool(hits)
        used.update(hits)
        if hits:
            n_trainable += p.numel()
    unused = set(prefixes) - used
    assert not unused, f"trainable prefixes match no parameter: {sorted(unused)}"
    return n_trainable


def freeze_bn_stats(model: nn.Module) -> None:
    """Put BatchNorm layers whose parameters are all frozen in eval mode (keeps ImageNet stats).

    Call after every ``model.train()``.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def param_groups(model: nn.Module, lrs: dict, weight_decay: float) -> list[dict]:
    """AdamW param groups, one lr per prefix; biases and norm weights (1-D) get no weight decay."""
    groups, seen = [], set()
    for prefix, lr in lrs.items():
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if p.requires_grad and name not in seen and matches(name, prefix):
                seen.add(name)
                (no_decay if p.ndim <= 1 else decay).append(p)
        assert decay or no_decay, f"lr prefix '{prefix}' matches no trainable parameter"
        groups.append({"params": decay, "lr": float(lr), "weight_decay": weight_decay, "name": prefix})
        groups.append({"params": no_decay, "lr": float(lr), "weight_decay": 0.0, "name": prefix})
    missing = [n for n, p in model.named_parameters() if p.requires_grad and n not in seen]
    assert not missing, f"trainable parameters without an lr: {missing[:3]} ..."
    return [g for g in groups if g["params"]]
