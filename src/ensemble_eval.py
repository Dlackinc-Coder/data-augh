"""Ensemble por promedio de probabilidades sobre el test split.

Uso:
  python -u src/ensemble_eval.py --cfg-a configs/seg_unet_a_v3.yaml \
      --ckpt-a checkpoints/exp_unet_a_v3/best.pt \
      --cfg-b configs/seg_dpt_swin_v3.yaml --ckpt-b checkpoints/exp_dpt_swin_v3/best.pt \
      --alpha 0.5 --outdir runs/ens_a3_g3 --save-1panel

Mismo protocolo de test que eval.py (tiles random 512, dice multiclase por batch).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import torch
from PIL import Image

from eval import (dice_multiclass, iou_multiclass, get_logits,
                  make_overlay, make_overlay_1panel)
from train_seg import load_cfg, PROJECT_ROOT
from dataset import make_loaders
from seg_model import build_model


def load_one(cfg_path, ckpt_path, device):
    cfg = load_cfg(cfg_path)
    model = build_model(cfg).to(device)
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-a", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--cfg-b", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--outdir", default="runs/ensemble")
    ap.add_argument("--save-1panel", action="store_true")
    ap.add_argument("--min-tissue", type=float, default=0.05)
    ap.add_argument("--max-overlays", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, ckpt_a = load_one(args.cfg_a, args.ckpt_a, device)
    model_b, ckpt_b = load_one(args.cfg_b, args.ckpt_b, device)
    cfg = load_cfg(args.cfg_a)
    _, _, test_dl = make_loaders(cfg)

    overlay_dir = os.path.join(PROJECT_ROOT, args.outdir)
    panel_dir = overlay_dir + "_1panel"
    os.makedirs(overlay_dir, exist_ok=True)
    if args.save_1panel:
        os.makedirs(panel_dir, exist_ok=True)
    for d in (overlay_dir, panel_dir if args.save_1panel else None):
        if d:
            for f in os.listdir(d):
                if f.endswith(".png"):
                    os.remove(os.path.join(d, f))

    dices, ious, candidates, bad = [], [], [], 0
    with torch.no_grad():
        for imgs, masks in test_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            la = get_logits(model_a, imgs).float()
            lb = get_logits(model_b, imgs).float()
            if not (torch.isfinite(la).all() and torch.isfinite(lb).all()):
                bad += 1
                continue
            pa = torch.softmax(la, dim=1)
            pb = torch.softmax(lb, dim=1)
            ens = args.alpha * pa + (1.0 - args.alpha) * pb
            log_ens = torch.log(ens + 1e-8)
            dices.append(dice_multiclass(log_ens, masks))
            ious.append(iou_multiclass(log_ens, masks))
            pred = ens.argmax(dim=1)
            for b in range(imgs.shape[0]):
                mask_np = masks[b].cpu().numpy()
                tissue = float((mask_np != 0).mean())
                if tissue >= args.min_tissue:
                    candidates.append((tissue, imgs[b].cpu(), mask_np,
                                       pred[b].cpu().numpy()))

    candidates.sort(key=lambda c: -c[0])
    for i, (tissue, img_t, gt, pred) in enumerate(candidates[:args.max_overlays]):
        Image.fromarray(make_overlay(img_t, gt, pred)).save(
            os.path.join(overlay_dir, f"overlay_{i:03d}.png"))
        if args.save_1panel:
            Image.fromarray(make_overlay_1panel(img_t, pred)).save(
                os.path.join(panel_dir, f"panel_{i:03d}.png"))

    d = sum(dices) / max(len(dices), 1)
    i = sum(ious) / max(len(ious), 1)
    name_a = os.path.basename(args.cfg_a).replace(".yaml", "")
    name_b = os.path.basename(args.cfg_b).replace(".yaml", "")
    print(f"TEST | dice {d:.4f} | iou {i:.4f} | ensemble "
          f"{name_a}*{args.alpha:.2f} + {name_b}*{1-args.alpha:.2f} "
          f"(ep {ckpt_a.get('epoch')}/{ckpt_b.get('epoch')})")
    if bad:
        print(f"AVISO: {bad} batches con logits no-finitos descartados")
    print(f"tiles con tejido >= {args.min_tissue:.0%}: {len(candidates)} "
          f"-> guardados {min(args.max_overlays, len(candidates))}")
    print(f"overlays 3 paneles: {overlay_dir}/")
    if args.save_1panel:
        print(f"overlays 1 panel: {panel_dir}/")


if __name__ == "__main__":
    main()
