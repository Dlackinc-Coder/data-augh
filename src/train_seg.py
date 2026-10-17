"""Training loop for semantic segmentation of mouse cardiac slices.

Features: AdamW/SGD with optional parameter groups (no weight decay for
attention biases), cosine/step LR schedule, AMP (fp16/bf16/fp32 with auto
detection), gradient accumulation and clipping, optional encoder freezing
for the first N epochs, NaN detection with rollback to the last valid state,
and checkpoint resume.
"""

import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import make_loaders
from seg_model import build_model, freeze_encoder, get_logits
from eval import dice_multiclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MAX_RESTORES = 3


def has_nan(model):
    """Check whether any floating parameter/buffer is non-finite."""
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
    for b in model.buffers():
        if b.is_floating_point() and not torch.isfinite(b).all():
            return True
    return False


def resolve(path):
    """Resolve a path relative to the project root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_cfg(path):
    """Load a YAML config and resolve dataset/checkpoint paths."""
    with open(resolve(path)) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["root"] = resolve(cfg["dataset"]["root"])
    ckpt = cfg.get("ckpt_dir") or (cfg.get("output") or {}).get("ckpt_dir") or "checkpoints"
    cfg["ckpt_dir"] = resolve(ckpt)
    if "ckpt_path" in cfg:
        cfg["ckpt_path"] = resolve(cfg["ckpt_path"])
    return cfg


def seed_everything(seed):
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, t):
    """Build optimizer with a no-weight-decay group for flagged parameters."""
    opt_name = str(t.get("optimizer", "adamw")).lower()
    lr = t["lr"]
    wd = t.get("weight_decay", 1e-4)

    no_decay_keys = ()
    if hasattr(model, "no_weight_decay_keywords"):
        try:
            no_decay_keys = tuple(model.no_weight_decay_keywords())
        except TypeError:
            no_decay_keys = ()

    if no_decay_keys:
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in no_decay_keys):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [{"params": decay, "weight_decay": wd},
                  {"params": no_decay, "weight_decay": 0.0}]
    else:
        groups = model.parameters()

    if opt_name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if opt_name == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, t, total_steps):
    """Build LR scheduler ('cosine', 'step' or 'none')."""
    sched_name = str(t.get("scheduler", "cosine")).lower()
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    if sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=t.get("step_size", 30), gamma=t.get("gamma", 0.1))
    if sched_name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {sched_name}")


def resolve_precision(t, use_cuda):
    """Return (precision, amp_dtype). 'auto' picks bf16 on capable GPUs."""
    precision = str(t.get("precision", "auto")).lower()
    if t.get("amp") is False:
        precision = "fp32"
    if precision == "auto":
        if use_cuda and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif use_cuda:
            precision = "fp16"
        else:
            precision = "fp32"
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"Unknown precision: {precision} (use fp16/bf16/fp32/auto)")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return precision, amp_dtype


def dice_ce_loss(logits, targets, dice_weight=1.0, ce_weight=1.0, class_weights=None,
                 label_smoothing=0.0, eps=1e-6):
    """Combined soft-Dice + CrossEntropy loss."""
    num_classes = logits.shape[1]

    weight = class_weights.to(logits.device) if class_weights is not None else None
    ce_loss = nn.functional.cross_entropy(
        logits, targets, weight=weight, label_smoothing=label_smoothing)

    probs = torch.softmax(logits, dim=1)
    dice_loss = 0.0
    for c in range(num_classes):
        p = probs[:, c]
        g = (targets == c).float()
        inter = (p * g).sum()
        dice_loss += 1 - (2 * inter + eps) / (p.sum() + g.sum() + eps)
    dice_loss /= num_classes

    return dice_weight * dice_loss + ce_weight * ce_loss


def main(cfg_path="configs/seg.yaml"):
    """Train a segmentation model defined by cfg_path (100+ lines of state)."""
    cfg = load_cfg(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.get("seed", 42))

    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = cfg.get("train", cfg.get("training", {})).get("log_dir", "runs/seg")
        writer = SummaryWriter(os.path.join(PROJECT_ROOT, log_dir))
    except ImportError:
        writer = None

    t = cfg.get("train", cfg.get("training", {}))
    train_dl, val_dl, _ = make_loaders(cfg, need_test=False)
    model = build_model(cfg).to(device)
    use_cuda = device.type == "cuda"

    loss_cfg = cfg.get("loss", {})
    label_smoothing = float(t.get("label_smoothing", 0.0))

    # Loss function
    loss_type = loss_cfg.get("type", "crossentropy")
    if loss_type == "dice_ce":
        dice_w = loss_cfg.get("dice_weight", 1.0)
        ce_w = loss_cfg.get("ce_weight", 1.0)
        class_weights = loss_cfg.get("class_weights", None)
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        def criterion(logits, targets):
            return dice_ce_loss(logits, targets, dice_w, ce_w, class_weights,
                                label_smoothing=label_smoothing)
    else:
        class_weights = loss_cfg.get("class_weights", None)
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=label_smoothing)

    ckpt_dir = cfg["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    state_path = os.path.join(ckpt_dir, "train_state.pt")

    # Transfer learning: freeze encoder for the first N epochs.
    freeze_epochs = cfg["model"].get("freeze_encoder_epochs", 0)

    precision, amp_dtype = resolve_precision(t, use_cuda)
    use_amp = use_cuda and precision != "fp32"
    scaler = torch.amp.GradScaler('cuda', enabled=(use_cuda and precision == "fp16"))
    accum = t.get("grad_accum", 1)
    grad_clip = t.get("grad_clip_norm", 1.0)

    if t.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  -> gradient checkpointing enabled")

    # Resume from the last saved state (model/optimizer/scheduler/scaler).
    resume = bool(t.get("resume", False))
    start_epoch = 1
    best_dice = 0.0
    st = None
    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=True)
        model.load_state_dict(st["model"])
        start_epoch = int(st["epoch"]) + 1
        best_dice = float(st.get("best_dice", 0.0))
        print(f"  -> Resuming from epoch {st['epoch']} (best_dice {best_dice:.4f})")

    if freeze_epochs > 0 and start_epoch <= freeze_epochs:
        freeze_encoder(model, freeze=True)
        print(f"  -> Encoder frozen for the first {freeze_epochs} epochs")

    optimizer = build_optimizer(model, t)
    total_steps = max(t["epochs"] * len(train_dl), 1)
    scheduler = build_scheduler(optimizer, t, total_steps)

    if st is not None:
        optimizer.load_state_dict(st["optimizer"])
        if scheduler is not None and st.get("scheduler"):
            scheduler.load_state_dict(st["scheduler"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])

    restores = 0
    for epoch in range(start_epoch, t["epochs"] + 1):
        # Unfreeze the encoder once the freeze window is over.
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            freeze_encoder(model, freeze=False)
            print(f"  -> Unfreezing encoder at epoch {epoch}")

        model.train()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for i, (imgs, masks) in enumerate(train_dl):
            imgs, masks = imgs.to(device), masks.to(device)
            target_size = masks.shape[-2:]
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits = get_logits(model, imgs, target_size=target_size)
                loss = criterion(logits, masks) / accum
            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                # Skip this micro-batch; keep previously accumulated gradients.
                continue
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_dl):
                if scaler.is_enabled():
                    # GradScaler skips the step itself on inf/nan gradients.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip,
                        error_if_nonfinite=False)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip,
                        error_if_nonfinite=False)
                    if torch.isfinite(total_norm):
                        optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            running += loss_val * accum

        model.eval()
        dices, val_loss = [], 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                target_size = masks.shape[-2:]
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    logits = get_logits(model, imgs, target_size=target_size)
                    val_loss += criterion(logits, masks).item()
                dices.append(dice_multiclass(logits.float(), masks))
        val_loss /= max(len(val_dl), 1)
        val_dice = sum(dices) / max(len(dices), 1)
        train_loss = running / len(train_dl)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else t["lr"]

        print(f"epoch {epoch:03d} | train {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_dice {val_dice:.4f} | "
              f"lr {lr:.2e} | {time.time()-t0:.1f}s")

        if has_nan(model):
            if restores >= MAX_RESTORES or not os.path.exists(state_path):
                raise RuntimeError("Divergencia a NaN repetida; abortando")
            restores += 1
            print(f"  !! NaN en params/buffers -> restaurando ultimo estado valido "
                  f"(restauracion {restores}/{MAX_RESTORES})")
            st = torch.load(state_path, map_location=device, weights_only=True)
            model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            if scheduler is not None and st.get("scheduler"):
                scheduler.load_state_dict(st["scheduler"])
            if st.get("scaler"):
                scaler.load_state_dict(st["scaler"])
            best_dice = st["best_dice"]
            continue

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_dice": best_dice}, state_path)

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metric/val_dice", val_dice, epoch)
            writer.add_scalar("lr", lr, epoch)

        torch.save({"model": model.state_dict(), "cfg": cfg,
                    "epoch": epoch, "dice": val_dice},
                   os.path.join(ckpt_dir, "last.pt"))
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "epoch": epoch, "dice": val_dice},
                       os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> best checkpoint (dice {best_dice:.4f})")

    if writer:
        writer.close()
    print(f"done. best val dice = {best_dice:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/seg.yaml")
