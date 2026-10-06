"""Train / eval transforms with torchvision.transforms.v2 (CLAUDE.md 3.5).

The CPU transforms work at the native ``source_size`` (48x48), which is ~5x cheaper
than augmenting at 224x224; ``resize_batch`` then upsamples the batch to ``img_size``
on the GPU. The source images are 48x48, so no information is lost.
"""
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(data_cfg: dict) -> v2.Normalize:
    ch = data_cfg["in_channels"]
    if data_cfg["normalize"] == "imagenet":
        assert ch == 3, "ImageNet normalization needs 3 input channels"
        return v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if data_cfg["normalize"] == "half":
        return v2.Normalize((0.5,) * ch, (0.5,) * ch)
    raise ValueError(f"Unknown normalize: {data_cfg['normalize']}")


def resize_batch(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Upsample a normalized (N, C, source_size, source_size) batch to the model input size."""
    size = cfg["data"]["img_size"]
    if x.shape[-2:] == (size, size):
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def build_transforms(cfg: dict, train: bool) -> v2.Compose:
    """Augmented train transform, or the val/test transform (also used when augment is off).

    Output size is ``source_size``; apply ``resize_batch`` on the device afterwards.
    """
    d, aug = cfg["data"], cfg["augment"]
    size = d["source_size"]
    if d["grayscale"]:
        # Images are loaded as 'L'; Grayscale(3) replicates them for the pretrained models.
        color = [v2.Grayscale(num_output_channels=d["in_channels"])]
    else:
        assert d["in_channels"] == 3, "grayscale: false needs in_channels: 3"
        color = []
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), _normalize(d)]

    if not (train and aug["enabled"]):
        return v2.Compose(color + [v2.Resize((size, size), antialias=True)] + to_tensor)

    assert aug["rotation"] <= 15, "CLAUDE.md 3.5: no rotations above 15 degrees"
    return v2.Compose(color + [
        v2.RandomResizedCrop(size, scale=tuple(aug["crop_scale"]), ratio=tuple(aug["crop_ratio"]),
                             antialias=True),
        v2.RandomHorizontalFlip(p=aug["hflip"]),
        v2.RandomRotation(degrees=aug["rotation"], interpolation=v2.InterpolationMode.BILINEAR),
        v2.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
    ] + to_tensor + [
        v2.RandomErasing(p=aug["erasing_p"], scale=tuple(aug["erasing_scale"])),
    ])
