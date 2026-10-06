"""Model factory shared by training, evaluation, Grad-CAM and the demo."""
from .cnn_scratch import EmotionCNN
from .pretrained import build_mobilenet_v2, build_resnet18

MODELS = ("cnn_scratch", "resnet18", "mobilenet_v2")


def build_model(name: str, num_classes: int, cfg: dict):
    """Build one of the three project models from its (merged) config."""
    m, d = cfg["model"], cfg["data"]
    if name == "cnn_scratch":
        return EmotionCNN(num_classes, in_channels=d["in_channels"], width=m["width"])
    assert d["in_channels"] == 3, f"{name} expects in_channels: 3"
    if name == "resnet18":
        return build_resnet18(num_classes, m["pretrained"], float(m["dropout"]))
    if name == "mobilenet_v2":
        return build_mobilenet_v2(num_classes, m["pretrained"], float(m["dropout"]))
    raise ValueError(f"Unknown model '{name}', choose from {MODELS}")
