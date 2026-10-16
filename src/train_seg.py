import os
import math
import re
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import make_loaders
from seg_model import build_model, freeze_encoder
from eval import dice_multiclass


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def has_nan(model):
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
    for b in model.buffers():
        if b.is_floating_point() and not torch.isfinite(b).all():
            return True
    return False


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_cfg(path):
    with open(resolve(path)) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["root"] = resolve(cfg["dataset"]["root"])
    ckpt = cfg.get("ckpt_dir") or (cfg.get("output") or {}).get("ckpt_dir") or "checkpoints"
    cfg["ckpt_dir"] = resolve(ckpt)
    if "ckpt_path" in cfg:
        cfg["ckpt_path"] = resolve(cfg["ckpt_path"])
    return cfg


def get_logits(model, imgs, target_size=None):
    """Extract logits from model output and optionally resize to target_size."""
    out = model(imgs)
    if hasattr(out, 'logits'):
        logits = out.logits  # HF SegFormer
    else:
        logits = out  # smp.Unet or raw tensor
    
    if target_size is not None and logits.shape[-2:] != target_size:
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
    return logits


def dice_ce_loss(logits, targets, dice_weight=1.0, ce_weight=1.0, class_weights=None, eps=1e-6):
    """Combined Dice + CrossEntropy loss."""
    num_classes = logits.shape[1]
    
    # CrossEntropy part
    if class_weights is not None:
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights.to(logits.device))
    else:
        ce_loss = F.cross_entropy(logits, targets)
    
    # Dice part (soft Dice)
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
    cfg = load_cfg(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = cfg.get("train", cfg.get("training", {})).get("log_dir", "runs/seg")
        writer = SummaryWriter(os.path.join(PROJECT_ROOT, log_dir))
    except ImportError:
        writer = None

    train_dl, val_dl, _ = make_loaders(cfg)
    model = build_model(cfg).to(device)
    use_cuda = device.type == "cuda"

    t = cfg.get("train", cfg.get("training", {}))
    loss_cfg = cfg.get("loss", {})
    
    # Loss function
    loss_type = loss_cfg.get("type", "crossentropy")
    if loss_type == "dice_ce":
        dice_w = loss_cfg.get("dice_weight", 1.0)
        ce_w = loss_cfg.get("ce_weight", 1.0)
        class_weights = loss_cfg.get("class_weights", None)
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        def criterion(logits, targets):
            return dice_ce_loss(logits, targets, dice_w, ce_w, class_weights)
    else:
        class_weights = loss_cfg.get("class_weights", None)
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    if t.get("llrd", False):
        # LLRD (layer-wise LR decay,15-oct-2026): capas profundas del encoder
        # con LR menor, decoder/cabecera con LR base. Para Swin via smp DPT los
        # stages del encoder se llaman encoder.model.layers.{0..3}.
        decay = t.get("llrd_decay", 0.75)
        groups = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            m = re.search(r"encoder\.model\.layers\.(\d+)\.", name)
            if m:
                k = int(m.group(1))
                lr = t["lr"] * (decay ** max(3 - k, 0))
            elif "encoder.model.patch_embed" in name:
                lr = t["lr"] * (decay ** 4)   # entrada del encoder (lo mas profundo)
            else:
                lr = t["lr"]                   # decoder, cabecera, norms top
            groups.setdefault(lr, []).append(p)
        param_groups = [{"params": ps, "lr": lr} for lr, ps in groups.items()]
        print("  -> LLRD:", [(f"{g['lr']:.2e}x{len(g['params'])}p") for g in param_groups])
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups, lr=t["lr"],
        weight_decay=t.get("weight_decay", 1e-4))
    
    precision = t.get("precision", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = use_cuda and precision != "fp32"
    scaler = torch.amp.GradScaler('cuda', enabled=(use_cuda and precision == "fp16"))
    accum = t.get("grad_accum", 1)

    # Scheduler (fix 15-oct-2026): scheduler.step() corre 1 de cada `accum`
    # micro-pasos, asi que T_max debe contener pasos de OPTIMIZADOR, no de batch.
    # Antes: T_max = epochs*len(train_dl) -> el coseno solo recorria 1/4 del arco
    # y el LR terminaba al 85% del pico (nunca anealaba).
    steps_per_epoch = max(math.ceil(len(train_dl) / accum), 1)
    total_steps = max(t["epochs"] * steps_per_epoch, 1)
    warmup_epochs = int(t.get("warmup_epochs", 0))
    if warmup_epochs > 0:
        warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps - 1)
        sched_warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps)
        sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [sched_warm, sched_cos], milestones=[warmup_steps])
        print(f"  -> scheduler: {warmup_epochs} ep warmup (0.01x->1x) + cosine "
              f"({total_steps} pasos optimizer, {steps_per_epoch}/ep)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        print(f"  -> scheduler: cosine {total_steps} pasos optimizer ({steps_per_epoch}/ep)")
    
    # Transfer learning: freeze encoder epochs
    freeze_epochs = cfg["model"].get("freeze_encoder_epochs", 0)
    encoder_frozen = False
    
    best_dice = 0.0
    ckpt_dir = cfg["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    state_path = os.path.join(ckpt_dir, "train_state.pt")
    restores = 0
    for epoch in range(1, t["epochs"] + 1):
        # aug v2: las ventanas de train dependen de la epoch (dataset.set_epoch)
        if hasattr(train_dl.dataset, "set_epoch"):
            train_dl.dataset.set_epoch(epoch)
        # Handle encoder freezing/unfreezing
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            freeze_encoder(model, freeze=False)
            print(f"  -> Unfreezing encoder at epoch {epoch}")
        
        model.train()
        t0 = time.time()
        optimizer.zero_grad()
        running = 0.0
        for i, (imgs, masks) in enumerate(train_dl):
            imgs, masks = imgs.to(device), masks.to(device)
            target_size = masks.shape[-2:]
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits = get_logits(model, imgs, target_size=target_size)
                loss = criterion(logits, masks) / accum
            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_dl):
                scaler.unscale_(optimizer)
                grads_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        p.grad.zero_()
                if grads_ok:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
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
                dices.append(dice_multiclass(logits, masks))
        val_loss /= max(len(val_dl), 1)
        val_dice = sum(dices) / max(len(dices), 1)
        train_loss = running / len(train_dl)
        lr = scheduler.get_last_lr()[0]

        print(f"epoch {epoch:03d} | train {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_dice {val_dice:.4f} | "
              f"lr {lr:.2e} | {time.time()-t0:.1f}s")

        if has_nan(model):
            restores += 1
            print(f"  !! NaN en params/buffers -> restaurando ultimo estado valido "
                  f"(restauracion {restores}/3)")
            if restores > 3 or not os.path.exists(state_path):
                raise RuntimeError("Divergencia a NaN repetida; abortando")
            st = torch.load(state_path, map_location=device, weights_only=False)
            model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            if st.get("scaler"):
                scaler.load_state_dict(st["scaler"])
            best_dice = st["best_dice"]
            continue

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
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
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/seg.yaml")
