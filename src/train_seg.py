import os
import time
import yaml
import torch
import torch.nn as nn

from dataset import make_loaders
from seg_model import build_model
from eval import dice_multiclass


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_cfg(path):
    with open(resolve(path)) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["root"] = resolve(cfg["dataset"]["root"])
    cfg["ckpt_dir"] = resolve(cfg.get("ckpt_dir", "checkpoints"))
    if "ckpt_path" in cfg:
        cfg["ckpt_path"] = resolve(cfg["ckpt_path"])
    return cfg


def main(cfg_path="configs/seg.yaml"):
    cfg = load_cfg(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(PROJECT_ROOT, "runs", "seg"))
    except ImportError:
        writer = None

    train_dl, val_dl, _ = make_loaders(cfg)
    model = build_model(cfg).to(device)
    use_cuda = device.type == "cuda"

    t = cfg["train"]
    criterion = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["lr"],
        weight_decay=t.get("weight_decay", 1e-4))
    total_steps = max(t["epochs"] * len(train_dl), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    accum = t.get("grad_accum", 1)
    best_dice = 0.0
    ckpt_dir = cfg["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0 = time.time()
        optimizer.zero_grad()
        running = 0.0
        for i, (imgs, masks) in enumerate(train_dl):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = model(imgs)
                loss = criterion(logits, masks) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
            running += loss.item() * accum

        model.eval()
        dices, val_loss = [], 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    logits = model(imgs)
                    val_loss += criterion(logits, masks).item()
                dices.append(dice_multiclass(logits, masks))
        val_loss /= max(len(val_dl), 1)
        val_dice = sum(dices) / max(len(dices), 1)
        train_loss = running / len(train_dl)
        lr = scheduler.get_last_lr()[0]

        print(f"epoch {epoch:03d} | train {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_dice {val_dice:.4f} | "
              f"lr {lr:.2e} | {time.time()-t0:.1f}s")
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
