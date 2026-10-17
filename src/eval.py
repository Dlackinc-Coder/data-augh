"""Evaluation for mouse cardiac slice segmentation.

Computes global (confusion-matrix based) macro Dice/IoU plus per-class
metrics on the test split, and saves visualization overlays for the tiles
with the most tissue (top-k bounded, so memory stays constant).
"""

import argparse
import heapq
import itertools
import os

import numpy as np
import torch
from PIL import Image

from seg_model import build_model, get_logits

CLASS_COLORS = np.array([
    [0, 0, 0],        # 0 background
    [0, 200, 0],      # 1 LV verde
    [0, 120, 255],    # 2 AAR azul
    [255, 0, 0],      # 3 IS rojo
], dtype=np.uint8)

OVERLAY_COLORS = np.array([
    [0, 0, 0],          # 0 background (invisible)
    [0, 200, 200],      # 1 LV viable cyan
    [255, 220, 0],      # 2 AAR riesgo amarillo
    [255, 40, 40],      # 3 IS infarto rojo
], dtype=np.uint8)

OVERLAY_ALPHA = [0.0, 0.35, 0.40, 0.50]

CLASS_NAMES = ["bg", "LV", "AAR", "IS"]


def dice_multiclass(logits, targets, eps=1e-6, ignore_index=None):
    """Mean Dice over classes present in the batch (skips empty classes)."""
    probs = torch.softmax(logits.float(), dim=1)
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
    """Mean IoU over classes present in the batch (skips empty classes)."""
    pred = logits.float().argmax(dim=1)
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


def per_class_metrics(cm, num_classes, eps=1e-6):
    """Per-class Dice/IoU from a KxK confusion matrix (rows=GT, cols=pred).

    Returns dict: class index -> {"dice", "iou", "support"} (None if absent).
    """
    out = {}
    for c in range(num_classes):
        tp = cm[c, c].item()
        fn = cm[c, :].sum().item() - tp
        fp = cm[:, c].sum().item() - tp
        support = cm[c, :].sum().item()
        if tp + fp + fn == 0:
            out[c] = None
            continue
        out[c] = {
            "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
            "iou": (tp + eps) / (tp + fp + fn + eps),
            "support": support,
        }
    return out


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


def make_overlay_1panel(img_t, pred_mask):
    orig = denorm_img(img_t).astype(np.float32)
    alpha_map = np.array(OVERLAY_ALPHA, dtype=np.float32)[pred_mask]
    color_map = OVERLAY_COLORS[np.clip(pred_mask, 0, 3)].astype(np.float32)
    out = orig * (1 - alpha_map[..., None]) + color_map * alpha_map[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _clean_dir(dir_path, prefix):
    """Remove only overlay files previously written by this script."""
    if not os.path.isdir(dir_path):
        return
    for f in os.listdir(dir_path):
        if f.startswith(prefix) and f.endswith(".png"):
            os.remove(os.path.join(dir_path, f))


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False, amp_dtype=torch.bfloat16,
             save_overlays=False, overlay_dir=None, save_1panel=False,
             panel_dir=None, max_overlays=64, min_tissue_ratio=0.05):
    """Evaluate a model on a loader.

    Returns dict with macro dice/iou, per-class metrics (confusion matrix
    based) and the number of candidate tiles found.
    """
    model.eval()
    num_classes = None
    cm = None
    candidate_heap = []
    counter = itertools.count()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            logits = get_logits(model, imgs, target_size=imgs.shape[-2:])
        logits = logits.float()
        if num_classes is None:
            num_classes = logits.shape[1]
            cm = torch.zeros(num_classes, num_classes, dtype=torch.float64)
        preds = logits.argmax(dim=1)
        flat = (num_classes * masks + preds).view(-1).cpu()
        cm += torch.bincount(flat, minlength=num_classes ** 2).reshape(
            num_classes, num_classes).to(torch.float64)

        if save_overlays or save_1panel:
            for b in range(imgs.shape[0]):
                mask_np = masks[b].cpu().numpy()
                tissue = float((mask_np != 0).mean())
                if tissue >= min_tissue_ratio:
                    item = (tissue, next(counter),
                            (imgs[b].cpu(), mask_np, preds[b].cpu().numpy()))
                    if len(candidate_heap) < max_overlays:
                        heapq.heappush(candidate_heap, item)
                    else:
                        heapq.heappushpop(candidate_heap, item)

    if cm is None:
        raise RuntimeError("Empty loader: nothing to evaluate")

    per_class = per_class_metrics(cm, num_classes)
    present = [v for v in per_class.values() if v is not None]
    macro_dice = float(np.mean([v["dice"] for v in present])) if present else 0.0
    macro_iou = float(np.mean([v["iou"] for v in present])) if present else 0.0

    candidates = sorted(candidate_heap, key=lambda c: -c[0])
    if save_overlays and overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        _clean_dir(overlay_dir, "overlay_")
    if save_1panel and panel_dir:
        os.makedirs(panel_dir, exist_ok=True)
        _clean_dir(panel_dir, "panel_")

    for i, (tissue, _, (img_t, gt, pred)) in enumerate(candidates[:max_overlays]):
        if save_overlays and overlay_dir:
            ov = make_overlay(img_t, gt, pred)
            Image.fromarray(ov).save(os.path.join(overlay_dir, f"overlay_{i:03d}.png"))
        if save_1panel and panel_dir:
            ov1 = make_overlay_1panel(img_t, pred)
            Image.fromarray(ov1).save(os.path.join(panel_dir, f"panel_{i:03d}.png"))

    return {
        "dice": macro_dice,
        "iou": macro_iou,
        "per_class": per_class,
        "n_candidates": len(candidate_heap),
        "num_classes": num_classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint")
    parser.add_argument("config", nargs="?", default="configs/seg.yaml")
    parser.add_argument("--save-1panel", action="store_true",
                        help="also save 1-panel alpha-blended overlays")
    parser.add_argument("--min-tissue", type=float, default=0.05,
                        help="minimum tissue ratio for a tile to be saved")
    parser.add_argument("--sampling", default=None,
                        help="override dataset sampling strategy (random|guided)")
    parser.add_argument("--ckpt", default=None, help="checkpoint path override")
    args = parser.parse_args()

    from train_seg import load_cfg, PROJECT_ROOT, resolve_precision
    from dataset import make_loaders

    cfg = load_cfg(args.config)
    if args.sampling:
        cfg.setdefault("dataset", {}).setdefault("sampling", {})["strategy"] = args.sampling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_dl = make_loaders(cfg)

    t = cfg.get("train", cfg.get("training", {}))
    precision, amp_dtype = resolve_precision(t, device.type == "cuda")
    use_amp = device.type == "cuda" and precision != "fp32"

    model = build_model(cfg).to(device)
    ckpt_path = args.ckpt or cfg.get(
        "ckpt_path", os.path.join(cfg["ckpt_dir"], "best.pt"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    max_overlays = cfg.get("eval", {}).get("max_overlays", 64)
    overlay_dir = os.path.join(PROJECT_ROOT, "runs", "test_overlays")
    panel_dir = os.path.join(PROJECT_ROOT, "runs", "test_overlays_1panel")
    res = evaluate(model, test_dl, device, use_amp=use_amp, amp_dtype=amp_dtype,
                   save_overlays=True, overlay_dir=overlay_dir,
                   save_1panel=args.save_1panel,
                   panel_dir=panel_dir if args.save_1panel else None,
                   max_overlays=max_overlays,
                   min_tissue_ratio=args.min_tissue)
    print(f"TEST | dice {res['dice']:.4f} | iou {res['iou']:.4f} "
          f"(ckpt epoch {ckpt.get('epoch')})")
    parts = []
    for c, v in res["per_class"].items():
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls{c}"
        if v is None:
            parts.append(f"{name}: n/a")
        else:
            parts.append(f"{name}: dice {v['dice']:.4f} iou {v['iou']:.4f} "
                         f"(n={int(v['support'])})")
    print("PER-CLASS | " + " | ".join(parts))
    print(f"tiles con tejido >= {args.min_tissue:.0%}: {res['n_candidates']} "
          f"-> guardados {min(max_overlays, res['n_candidates'])}")
    print(f"overlays 3 paneles: {overlay_dir}/")
    if args.save_1panel:
        print(f"overlays 1 panel: {panel_dir}/")


if __name__ == "__main__":
    main()
