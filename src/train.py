"""Train one model from a config file; the same loop serves all three models.

Examples:
    python -m src.train --config configs/cnn_scratch.yaml --seed 42 --dry-run
    python -m src.train --config configs/cnn_scratch.yaml --seed 42 --overfit-batch
    python -m src.train --config configs/resnet18.yaml --seed 42
"""
import argparse
import math
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FERDataset, PrefetchLoader, class_names, read_split
from src.data.transforms import build_transforms, resize_batch
from src.models import build_model
from src.models.pretrained import freeze_bn_stats, param_groups, set_trainable
from src.utils.config import load_config, save_config
from src.utils.logger import CSVLogger, plot_curves
from src.utils.metrics import classification_metrics
from src.utils.seed import seed_everything, seed_worker

DRY_RUN_BATCHES = 2

# AMP's GradScaler skips an optimizer step when the initial loss scale overflows; harmless.
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")


def make_optimizer(model, stage: dict, train_cfg: dict, device):
    """AdamW over the stage's lr groups (fused kernel on CUDA: fewer launches per step)."""
    groups = param_groups(model, stage["lr"], float(train_cfg["weight_decay"]))
    return torch.optim.AdamW(groups, fused=device.type == "cuda")


class FocalLoss(nn.Module):
    """Cross-entropy (with label smoothing / class weights) scaled by (1 - p_t)^gamma."""

    def __init__(self, gamma: float, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma, self.label_smoothing = gamma, label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = logits.softmax(dim=1).gather(1, target[:, None]).squeeze(1)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_class_weights(labels: np.ndarray, num_classes: int, train_cfg: dict):
    """w_c ∝ 1/sqrt(n_c) normalized to mean 1, or None when off (CLAUDE.md 4)."""
    mode = train_cfg["class_weight"]
    if mode not in ("auto", "none"):
        raise ValueError(f"class_weight must be 'auto' or 'none', got {mode!r}")
    counts = np.bincount(labels, minlength=num_classes)
    ratio = counts.max() / max(counts.min(), 1)
    if mode == "none" or ratio <= float(train_cfg["class_weight_ratio"]):
        return None
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_criterion(train_cfg: dict, weight):
    """Label-smoothed cross-entropy, or focal loss when ``focal_loss`` is on."""
    smoothing = float(train_cfg["label_smoothing"])
    if train_cfg["focal_loss"]:
        return FocalLoss(float(train_cfg["focal_gamma"]), weight, smoothing)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)


def warmup_cosine(optimizer, warmup_iters: int, total_iters: int, min_lr: float):
    """Per-iteration schedule: linear warmup, then cosine decay to ``min_lr`` in every group."""
    def make(base_lr: float):
        floor = min(min_lr / base_lr, 1.0)

        def factor(it: int) -> float:
            if it < warmup_iters:
                return (it + 1) / warmup_iters
            t = min((it - warmup_iters) / max(total_iters - warmup_iters, 1), 1.0)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
        return factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [make(g["lr"]) for g in optimizer.param_groups])


def make_loader(ds, batch_size: int, shuffle: bool, cfg: dict, device, seed: int, drop_last=False):
    """Seeded DataLoader, prefetched in a background thread (workers persist if num_workers > 0)."""
    workers = cfg["data"]["num_workers"]
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                        num_workers=workers, persistent_workers=workers > 0,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=gen)
    return PrefetchLoader(loader)


def autocast(device, enabled: bool):
    """float16 autocast on CUDA; a no-op context when disabled."""
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device.type, dtype=dtype, enabled=enabled)


def run_epoch(model, loader, criterion, cfg: dict, device, use_amp: bool, desc: str,
              optimizer=None, scheduler=None, scaler=None) -> dict:
    """One pass over ``loader``; trains when an optimizer is given. Returns loss, acc, macro-F1."""
    training = optimizer is not None
    model.train(training)
    if training:
        freeze_bn_stats(model)
        params = [p for p in model.parameters() if p.requires_grad]
    loss_sum = torch.zeros((), device=device)
    preds, targets = [], []
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty()):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = resize_batch(x, cfg)
            with autocast(device, use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, float(cfg["train"]["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            loss_sum += loss.detach().float() * y.size(0)
            preds.append(logits.argmax(1))
            targets.append(y)
    y_true = torch.cat(targets).cpu().numpy()
    y_pred = torch.cat(preds).cpu().numpy()
    m = classification_metrics(y_true, y_pred, cfg["data"]["num_classes"])
    return {"loss": loss_sum.item() / len(y_true), "acc": m["acc"], "macro_f1": m["macro_f1"]}


def make_run_dir(cfg: dict, seed: int, dry_run: bool) -> Path:
    """``runs/{model}_s{seed}_{YYYYmmdd-HHMM}/``; ``runs/dryrun_{model}/`` is overwritten each dry run."""
    root, name = Path(cfg["runs_dir"]), cfg["model"]["name"]
    if dry_run:
        run_dir = root / f"dryrun_{name}"
        shutil.rmtree(run_dir, ignore_errors=True)
    else:
        run_dir = base = root / f"{name}_s{seed}_{datetime.now():%Y%m%d-%H%M}"
        k = 1
        while run_dir.exists():
            k += 1
            run_dir = base.with_name(f"{base.name}-{k}")
    run_dir.mkdir(parents=True)
    return run_dir


def save_checkpoint(path, model, cfg: dict, classes: list[str], **info) -> None:
    """Weights plus everything needed to rebuild the model (config, class names)."""
    torch.save({"model": model.state_dict(), "model_name": cfg["model"]["name"],
                "classes": classes, "cfg": cfg, **info}, path)


def overfit_batch(cfg, model, train_df, criterion, device, use_amp: bool, steps: int, seed: int) -> bool:
    """Fit one fixed batch (no augmentation) with the last stage's setup; accuracy must reach ~100%."""
    t = cfg["train"]
    batch = train_df.sample(n=t["batch_size"], random_state=seed)
    ds = FERDataset(batch, build_transforms(cfg, train=False), cfg["data"]["grayscale"])
    x = resize_batch(torch.stack([ds[i][0] for i in range(len(ds))]).to(device), cfg)
    y = torch.as_tensor(ds.labels).to(device)

    stage = t["stages"][-1]
    set_trainable(model, stage["trainable"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, stage, t, device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"overfit-batch: {len(ds)} images, stage '{stage['name']}', {steps} steps, constant lr")
    for step in range(1, steps + 1):
        model.train()
        freeze_bn_stats(model)
        with autocast(device, use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, float(t["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 25 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  step {step:4d}  loss {loss.item():.4f}  acc (train mode) {acc:.3f}")
    model.eval()
    with torch.no_grad(), autocast(device, use_amp):
        acc = (model(x).argmax(1) == y).float().mean().item()
    ok = acc >= 0.99
    print(f"overfit-batch {'PASS' if ok else 'FAIL'}: accuracy on the batch (eval mode) = {acc:.3f}")
    return ok


def parse_args():
    ap = argparse.ArgumentParser(description="Train a FER model from a config file.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, help="default: first entry of `seeds` in the config")
    ap.add_argument("--dry-run", action="store_true",
                    help=f"{DRY_RUN_BATCHES} train and {DRY_RUN_BATCHES} val batches per stage")
    ap.add_argument("--overfit-batch", action="store_true",
                    help="train on one fixed batch; accuracy must approach 100%%")
    ap.add_argument("--overfit-steps", type=int, default=200)
    ap.add_argument("--cpu", action="store_true", help="use the CPU even if CUDA is available")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    cfg["seed"] = seed
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    torch.backends.cudnn.benchmark = device.type == "cuda"
    # The loader thread and the kernel-launching main thread share the GIL; finer switching
    # lets the main thread get it back sooner (~5-8% faster epochs in our measurements).
    sys.setswitchinterval(2e-4)
    d, t = cfg["data"], cfg["train"]
    assert t["optimizer"] == "adamw", "only AdamW is implemented"
    num_classes, name = d["num_classes"], cfg["model"]["name"]

    train_df, val_df = read_split(cfg, "train"), read_split(cfg, "val")
    classes = class_names(train_df, num_classes)
    assert class_names(val_df, num_classes) == classes, "train and val class folders differ"
    weight = compute_class_weights(train_df["label_idx"].to_numpy(), num_classes, t)
    criterion = build_criterion(t, weight).to(device)
    model = build_model(name, num_classes, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params:,} params | device {device} | AMP {use_amp} | seed {seed}")
    print(f"train {len(train_df):,} | val {len(val_df):,} | class weights "
          f"{'off' if weight is None else [round(w, 3) for w in weight.tolist()]}")

    if args.overfit_batch:
        ok = overfit_batch(cfg, model, train_df, criterion, device, use_amp, args.overfit_steps, seed)
        sys.exit(0 if ok else 1)

    if args.dry_run:
        train_df = train_df.sample(n=DRY_RUN_BATCHES * t["batch_size"], random_state=seed)
        val_df = val_df.sample(n=DRY_RUN_BATCHES * 2 * t["batch_size"], random_state=seed)

    start = time.time()
    train_ds = FERDataset(train_df, build_transforms(cfg, train=True), d["grayscale"], d["cache_in_memory"])
    val_ds = FERDataset(val_df, build_transforms(cfg, train=False), d["grayscale"], d["cache_in_memory"])
    if d["cache_in_memory"]:
        print(f"cached {len(train_ds) + len(val_ds):,} images in RAM ({time.time() - start:.1f}s)")
    train_loader = make_loader(train_ds, t["batch_size"], True, cfg, device, seed, drop_last=True)
    val_loader = make_loader(val_ds, 2 * t["batch_size"], False, cfg, device, seed)

    run_dir = make_run_dir(cfg, seed, args.dry_run)
    cfg["run"] = {
        "dir": run_dir.as_posix(), "dry_run": args.dry_run, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "amp": use_amp,
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_params": n_params, "classes": classes,
    }
    save_config(cfg, run_dir / "config.yaml")
    logger = CSVLogger(run_dir / "log.csv")
    print(f"run dir: {run_dir}")

    monitor = t["monitor"]
    sign = -1.0 if "loss" in monitor else 1.0          # maximize F1/acc, minimize loss
    best, epoch, stages = -math.inf, 0, t["stages"]
    for si, stage in enumerate(stages):
        n_trainable = set_trainable(model, stage["trainable"])
        optimizer = make_optimizer(model, stage, t, device)
        iters = len(train_loader)
        scheduler = warmup_cosine(optimizer, int(float(t["warmup_epochs"]) * iters),
                                  stage["epochs"] * iters, float(t["min_lr"]))
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        early_stop = si == len(stages) - 1             # scratch / Stage B only (CLAUDE.md 4)
        epochs = 1 if args.dry_run else stage["epochs"]
        bad = 0
        print(f"stage '{stage['name']}': {epochs} epoch(s) x {iters} iters | "
              f"trainable {n_trainable:,} params | lr {stage['lr']}")
        for stage_epoch in range(1, epochs + 1):
            epoch += 1
            tic = time.time()
            tr = run_epoch(model, train_loader, criterion, cfg, device, use_amp, f"train {epoch}",
                           optimizer, scheduler, scaler)
            va = run_epoch(model, val_loader, criterion, cfg, device, use_amp, f"val {epoch}")
            row = {
                "epoch": epoch, "stage": stage["name"], "stage_epoch": stage_epoch,
                "train_loss": tr["loss"], "train_acc": tr["acc"], "train_macro_f1": tr["macro_f1"],
                "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
                "lr": max(g["lr"] for g in optimizer.param_groups), "epoch_time": time.time() - tic,
            }
            logger.log(row)
            info = {"epoch": epoch, "stage": stage["name"], monitor: row[monitor]}
            improved = sign * row[monitor] > best
            if improved:
                best, bad = sign * row[monitor], 0
                save_checkpoint(run_dir / "best.pt", model, cfg, classes, **info)
            else:
                bad += 1
            save_checkpoint(run_dir / "last.pt", model, cfg, classes, **info)
            print(f"epoch {epoch:3d} [{stage['name']}] "
                  f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} F1 {tr['macro_f1']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['acc']:.4f} F1 {va['macro_f1']:.4f} | "
                  f"lr {row['lr']:.2e} | {row['epoch_time']:.0f}s{' *' if improved else ''}")
            if early_stop and bad >= int(t["early_stop_patience"]):
                print(f"early stopping: no {monitor} improvement for {bad} epochs")
                break

    plot_curves(run_dir / "log.csv", run_dir / "curves.png", title=run_dir.name)
    print(f"done: best {monitor} = {sign * best:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
