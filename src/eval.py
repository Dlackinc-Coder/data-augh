import os
import torch
import numpy as np
from PIL import Image

CLASS_COLORS = np.array([
    [0, 0, 0],        # 0 background
    [0, 200, 0],      # 1 LV verde
    [0, 120, 255],    # 2 AAR azul
    [255, 0, 0],      # 3 IS rojo
], dtype=np.uint8)


def dice_multiclass(logits, targets, eps=1e-6, ignore_index=None):
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    num_classes = probs.shape[1]
    dices = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        p = (pred == c).float()
        g = (targets == c).float()
        if p.sum() + g.sum() == 0:
            continue
        inter = (p * g).sum()
        dices.append((2 * inter + eps) / (p.sum() + g.sum() + eps))
    if not dices:
        return 0.0
    return torch.stack(dices).mean().item()


def iou_multiclass(logits, targets, eps=1e-6):
    pred = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    ious = []
    for c in range(num_classes):
        p = (pred == c).float()
        g = (targets == c).float()
        if p.sum() + g.sum() == 0:
            continue
        ious.append(((p * g).sum()) / ((p + g).sum() - (p * g).sum() + eps))
    if not ious:
        return 0.0
    return torch.stack(ious).mean().item()


def mask_to_color(mask):
    mask = mask.astype(np.int64)
    return CLASS_COLORS[np.clip(mask, 0, len(CLASS_COLORS) - 1)]


def denorm_img(img_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = img_t.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = (img * np.array(std) + np.array(mean)) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def make_overlay(img_t, gt_mask, pred_mask, pad=4):
    h, w = gt_mask.shape
    orig = denorm_img(img_t)
    gt_c = mask_to_color(gt_mask)
    pred_c = mask_to_color(pred_mask)
    sep = np.full((h, pad, 3), 255, dtype=np.uint8)
    row = np.concatenate([orig, sep, gt_c, sep, pred_c], axis=1)
    return row


@torch.no_grad()
def evaluate(model, loader, device, half=True, save_overlays=False,
             overlay_dir=None, max_overlays=64):
    model.eval()
    dices, ious = [], []
    idx = 0
    if save_overlays and overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.cuda.amp.autocast(enabled=half):
            logits = model(imgs)
        dices.append(dice_multiclass(logits.float(), masks))
        ious.append(iou_multiclass(logits.float(), masks))
        if save_overlays and overlay_dir:
            preds = logits.float().argmax(dim=1)
            for b in range(imgs.shape[0]):
                if idx >= max_overlays:
                    break
                ov = make_overlay(imgs[b], masks[b].cpu().numpy(),
                                  preds[b].cpu().numpy())
                Image.fromarray(ov).save(
                    os.path.join(overlay_dir, f"overlay_{idx:03d}.png"))
                idx += 1
    return sum(dices) / len(dices), sum(ious) / len(ious)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from train_seg import load_cfg, PROJECT_ROOT
    from dataset import make_loaders
    from seg_model import build_model

    cfg = load_cfg(sys.argv[1] if len(sys.argv) > 1 else "configs/seg.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_dl = make_loaders(cfg)
    model = build_model(cfg).to(device)
    ckpt = torch.load(cfg.get("ckpt_path",
                              os.path.join(cfg["ckpt_dir"], "best.pt")),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    overlay_dir = os.path.join(PROJECT_ROOT, "runs", "test_overlays")
    d, i = evaluate(model, test_dl, device,
                    save_overlays=True, overlay_dir=overlay_dir)
    print(f"TEST | dice {d:.4f} | iou {i:.4f} "
          f"(ckpt epoch {ckpt.get('epoch')})")
    print(f"overlays guardados en: {overlay_dir}/")
