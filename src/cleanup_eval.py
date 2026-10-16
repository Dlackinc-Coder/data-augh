#!/usr/bin/env python
"""cleanup_eval.py — reevaluacion offline de preds guardadas (+ limpieza).

Lee <run-dir>/preds/<stem>.png (uint8 0..3, guardadas por src/eval_fullimage.py
con --save-preds 1) y las mascaras GT del split (config indicada en el
metrics.json del run), aplica cleanup_pred para cada variante de la rejilla y
recalcula las metricas con evaluate_pair/aggregate_per_image (convenciones
identicas a eval_fullimage: misma clase ausente -> omitida; GT presente y pred
ausente -> dice 0; micro con intersecciones).

Uso:
  python -u src/cleanup_eval.py --run-dir runs/eval_honest_val/G-v2 \
      --grid "0,0," --grid "2,2000," --grid "2,4000,128" \
      --out runs/eval_cleanup_val/G-v2.json

  # la variante "0,0," (sin limpieza) debe reproducir el metrics.json del run
  # (verificacion no-op del pipeline de agregacion).

Formato de --grid: "close_r,min_px,max_dist" (max_dist vacio/None = sin poda
por distancia). Repetible.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import cv2
import numpy as np

from train_seg import load_cfg, PROJECT_ROOT
from eval_fullimage import find_pairs, evaluate_pair, aggregate_per_image
from cleanup import cleanup_pred


def parse_variant(spec):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"grid invalida {spec!r}: formato 'close_r,min_px,max_dist'")
    close_r, min_px = int(parts[0]), int(parts[1])
    md = parts[2]
    max_dist = None if md in ("", "none", "None", "-1") else float(md)
    label = (f"cl{close_r}_min{min_px}_"
             f"d{'none' if max_dist is None else int(max_dist)}")
    return {"label": label, "close_r": close_r, "min_cluster_px": min_px,
            "max_dist": max_dist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--grid", action="append", default=None,
                    help="variante 'close_r,min_px,max_dist' (repetible); def: sin limpieza")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run_dir = (args.run_dir if os.path.isabs(args.run_dir)
               else os.path.join(PROJECT_ROOT, args.run_dir))
    with open(os.path.join(run_dir, "metrics.json")) as f:
        meta = json.load(f)
    split = args.split or meta.get("split", "test")
    cfg = load_cfg(meta["cfg"])
    root = args.root or cfg["dataset"]["root"]
    if not os.path.isabs(root):
        root = os.path.join(PROJECT_ROOT, root)
    pairs = find_pairs(root, split)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    variants = [parse_variant(s) for s in (args.grid or ["0,0,"])]

    gts = {}
    for stem, _img, mask_path in pairs:
        g = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gts[stem] = np.clip(np.rint(g.astype(np.float32)), 0, 3).astype(np.uint8)

    preds_dir = os.path.join(run_dir, "preds")
    baseline = (meta.get("aggregate") or {}).get("avg3_mean")
    base_s = "—" if baseline is None else f"{baseline * 100:.2f}"
    print(f"[cleanup_eval] {os.path.relpath(run_dir, PROJECT_ROOT)} | split {split} | "
          f"{len(pairs)} pares | baseline avg3 {base_s}", flush=True)

    def fmt(v):
        return "—" if v is None else f"{v * 100:.2f}"

    results = []
    for v in variants:
        t0 = time.time()
        per_image = {}
        missing = 0
        for stem, _img, _mask in pairs:
            pred = cv2.imread(os.path.join(preds_dir, f"{stem}.png"),
                              cv2.IMREAD_GRAYSCALE)
            if pred is None:
                missing += 1
                continue
            pred = np.clip(np.rint(pred.astype(np.float32)), 0, 3).astype(np.uint8)
            if v["close_r"] or v["min_cluster_px"] or v["max_dist"] is not None:
                pred = cleanup_pred(pred, close_r=v["close_r"],
                                    min_cluster_px=v["min_cluster_px"],
                                    max_dist=v["max_dist"])
            rec, _comps, _n = evaluate_pair(pred, gts[stem], want_slice=False)
            per_image[stem] = rec
        agg = aggregate_per_image(per_image)
        d = agg["dice_mean"]
        hd95 = "—" if agg["avg_hd95"] is None else f"{agg['avg_hd95']:.1f}"
        hd = "—" if agg["avg_hd"] is None else f"{agg['avg_hd']:.1f}"
        print(f"[{v['label']}] n={len(per_image)} avg3={fmt(agg['avg3_mean'])} "
              f"LV={fmt(d[1])} AAR={fmt(d[2])} IS={fmt(d[3])} "
              f"micro={fmt(agg['avg3_micro'])} HD95={hd95} HD={hd} "
              f"({time.time() - t0:.0f}s, faltantes={missing})", flush=True)
        results.append({"variant": v, "n_images": len(per_image),
                        "missing_preds": missing, "aggregate": agg,
                        "elapsed_s": round(time.time() - t0, 1)})

    out_path = (args.out if os.path.isabs(args.out)
                else os.path.join(PROJECT_ROOT, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"run_dir": os.path.relpath(run_dir, PROJECT_ROOT),
                   "split": split, "baseline_avg3_mean": baseline,
                   "preds_dir": os.path.relpath(preds_dir, PROJECT_ROOT),
                   "n_pairs": len(pairs), "results": results,
                   "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    print(f"[cleanup_eval] -> {os.path.relpath(out_path, PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
