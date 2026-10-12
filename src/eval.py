import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

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


def make_overlay_1panel(img_t, pred_mask):
    orig = denorm_img(img_t).astype(np.float32)
    alpha_map = np.array(OVERLAY_ALPHA, dtype=np.float32)[pred_mask]
    color_map = OVERLAY_COLORS[np.clip(pred_mask, 0, 3)].astype(np.float32)
    out = orig * (1 - alpha_map[..., None]) + color_map * alpha_map[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def get_logits(model, imgs):
    """Extract logits from model output (handles different model types)."""
    out = model(imgs)
    if hasattr(out, 'logits'):
        logits = out.logits  # HF SegFormer
    else:
        logits = out  # smp.Unet or raw tensor
    if logits.shape[-2:] != imgs.shape[-2:]:
        logits = F.interpolate(logits, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def evaluate(model, loader, device, half=True, save_overlays=False,
             overlay_dir=None, save_1panel=False, panel_dir=None,
             max_overlays=64, min_tissue_ratio=0.05):
    model.eval()
    dices, ious = [], []
    candidates = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast('cuda', enabled=half):
            logits = get_logits(model, imgs)
        dices.append(dice_multiclass(logits.float(), masks))
        ious.append(iou_multiclass(logits.float(), masks))
        if save_overlays or save_1panel:
            preds = logits.float().argmax(dim=1)
            for b in range(imgs.shape[0]):
                mask_np = masks[b].cpu().numpy()
                tissue = float((mask_np != 0).mean())
                if tissue >= min_tissue_ratio:
                    candidates.append(
                        (tissue, imgs[b].cpu(), mask_np,
                         preds[b].cpu().numpy()))

    if save_overlays and overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        for f in os.listdir(overlay_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(overlay_dir, f))
    if save_1panel and panel_dir:
        os.makedirs(panel_dir, exist_ok=True)
        for f in os.listdir(panel_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(panel_dir, f))

    candidates.sort(key=lambda c: -c[0])
    for i, (tissue, img_t, gt, pred) in enumerate(candidates[:max_overlays]):
        if save_overlays and overlay_dir:
            ov = make_overlay(img_t, gt, pred)
            Image.fromarray(ov).save(
                os.path.join(overlay_dir, f"overlay_{i:03d}.png"))
        if save_1panel and panel_dir:
            ov1 = make_overlay_1panel(img_t, pred)
            Image.fromarray(ov1).save(
                os.path.join(panel_dir, f"panel_{i:03d}.png"))

    return sum(dices) / len(dices), sum(ious) / len(ious), len(candidates)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from train_seg import load_cfg, PROJECT_ROOT
    from dataset import make_loaders
    from seg_model import build_model

    args = sys.argv[1:]
    save_1panel = "--save-1panel" in args
    args = [a for a in args if a != "--save-1panel"]
    min_tissue = 0.05
    sampling_override = None
    ckpt_override = None
    for a in list(args):
        if a.startswith("--min-tissue="):
            min_tissue = float(a.split("=")[1])
            args.remove(a)
        elif a.startswith("--sampling="):
            sampling_override = a.split("=")[1]
            args.remove(a)
        elif a.startswith("--ckpt="):
            ckpt_override = a.split("=", 1)[1]
            args.remove(a)

    cfg = load_cfg(args[0] if args else "configs/seg.yaml")
    if sampling_override:
        cfg.setdefault("dataset", {}).setdefault("sampling", {})["strategy"] = sampling_override
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_dl = make_loaders(cfg)
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_override or cfg.get("ckpt_path",
                              os.path.join(cfg["ckpt_dir"], "best.pt")),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    overlay_dir = os.path.join(PROJECT_ROOT, "runs", "test_overlays")
    panel_dir = os.path.join(PROJECT_ROOT, "runs", "test_overlays_1panel")
    d, i, n_cand = evaluate(model, test_dl, device,
                             save_overlays=True, overlay_dir=overlay_dir,
                             save_1panel=save_1panel,
                             panel_dir=panel_dir if save_1panel else None,
                             min_tissue_ratio=min_tissue)
    print(f"TEST | dice {d:.4f} | iou {i:.4f} "
          f"(ckpt epoch {ckpt.get('epoch')})")
    print(f"tiles con tejido >= {min_tissue:.0%}: {n_cand} "
          f"-> guardados {min(64, n_cand)}")
    print(f"overlays 3 paneles: {overlay_dir}/")
    if save_1panel:
        print(f"overlays 1 panel: {panel_dir}/")
