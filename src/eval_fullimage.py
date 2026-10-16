#!/usr/bin/env python
"""eval_fullimage.py — Evaluacion HONESTA full-image (Fase 2, 16-oct-2026).

Protocolo honesto (a diferencia del protocolo "random tiles" de eval.py):
  - ventana deslizante sobre la imagen COMPLETA (6000x4000) con solape
  - mezcla Hann de las probabilidades acumuladas
  - prediccion final por argmax de la probabilidad media por pixel

Metricas por clase de tejido (LV / AAR / IS), comparables con DUCNet 2026:
  - Dice e IoU por imagen
      * clase ausente en GT y en pred  -> se OMITE
      * presente en GT pero no predicha -> Dice = 0 (cuenta en la media)
      * ausente en GT pero predicha     -> no entra en la media; se cuenta
        aparte como imagen con falso positivo (y entra en la metrica micro)
  - HD y HD95 (Hausdorff, superficie) por imagen y clase
  - ratio de tamano de infarto (IS/LV, IS/AAR): GT vs pred (media/imprecision)
  - media de las 3 clases de tejido (formato de la tabla de DUCNet: 80.68)
  - metrica secundaria por "rodaja" (componentes conexas del GT): aproxima
    la evaluacion por slice del paper

Salidas:
  <outdir>/metrics.json   (todo, por imagen + agregados)
  <outdir>/overlays/*.png (N imagenes con pred vs GT, reducidas)
  resumen tabular por stdout

Uso:
  python -u src/eval_fullimage.py \
      --cfg configs/seg_dpt_swin_v2.yaml \
      --ckpt checkpoints/exp_dpt_swin_v2/best.pt \
      --outdir runs/eval_honest/G-v2 --tag G-v2 \
      --amp bf16 --batch 8 --save-overlays 6

  # ensemble de dos modelos (media de probabilidades, peso alpha al modelo A):
  python -u src/eval_fullimage.py --cfg configs/seg_unet_a_v3.yaml \
      --ckpt checkpoints/exp_unet_a_v3/best.pt \
      --cfg-b configs/seg_dpt_swin_v3.yaml \
      --ckpt-b checkpoints/exp_dpt_swin_v3/best.pt --alpha 0.5 \
      --outdir runs/eval_honest/A-v3+G-v3 --tag "A-v3+G-v3"

Opciones: --tile (def. cfg/512), --stride (def. tile/2), --amp fp32|bf16,
  --tta none|flips|d4, --max-images N (0=todas), --save-overlays N,
  --save-preds 0|1 (def. 1: guarda outdir/preds/<stem>.png),
  --split test|val (def. test), --device auto|cuda|cpu
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import numpy as np
import cv2
import torch

from train_seg import load_cfg, PROJECT_ROOT
from seg_model import build_model
from eval import get_logits

FG_CLASSES = [1, 2, 3]
CLS_NAMES = {0: "bg", 1: "LV", 2: "AAR", 3: "IS"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_COLORS = np.array(
    [[0, 0, 0], [0, 200, 0], [0, 120, 255], [255, 0, 0]], dtype=np.uint8)


# ----------------------------------------------------------------------------
# utilidades de ventana / blending
# ----------------------------------------------------------------------------
def window_starts(length, tile, stride):
    """Posiciones de inicio de ventana que cubren [0, length) por completo."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def hann2d(tile, min_w=1e-3):
    w1 = np.hanning(tile).astype(np.float32)
    w = np.outer(w1, w1)
    return np.clip(w, min_w, None)


def _tta_transforms(kind):
    """Devuelve [(nombre, fwd, inv)] para test-time augmentation."""
    if kind == "none":
        return [("id", lambda t: t, lambda t: t)]
    flips = [
        ("id", lambda t: t, lambda t: t),
        ("h", lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
        ("v", lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
        ("hv", lambda t: torch.flip(t, [2, 3]), lambda t: torch.flip(t, [2, 3])),
    ]
    if kind == "flips":
        return flips
    if kind == "d4":
        out = []
        base = [("id", lambda t: t), ("h", lambda t: torch.flip(t, [3]))]
        for k in range(4):
            for name, f in base:
                fwd = (lambda t, k=k, f=f: f(torch.rot90(t, k, [2, 3])))
                inv = (lambda t, k=k, f=f: torch.rot90(f(t), -k, [2, 3]))
                out.append((f"{name}{k}", fwd, inv))
        return out
    raise ValueError(f"tta desconocido: {kind}")


class NonFiniteLogits(Exception):
    """Logits con NaN/Inf durante la eval (run corrupto, no reintentar)."""


def _probs(model, xt, amp_dtype):
    if amp_dtype is not None and xt.is_cuda:
        with torch.no_grad(), torch.autocast("cuda", enabled=True, dtype=amp_dtype):
            logits = get_logits(model, xt)
    else:
        with torch.no_grad():
            logits = get_logits(model, xt)
    logits = logits.float()
    if not torch.isfinite(logits).all():
        return None
    return torch.softmax(logits, dim=1)


def infer_full_image(model, model_b=None, alpha=0.5, img_rgb=None, tile=512,
                     stride=256, batch=8, device="cuda", amp_dtype=None,
                     tta="none", verbose=True):
    """Inferencia ventana deslizante + mezcla Hann. Devuelve pred uint8 HxW."""
    H, W = img_rgb.shape[:2]
    if H < tile or W < tile:
        raise ValueError(f"imagen {H}x{W} menor que tile {tile}")
    x = (img_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    ys = window_starts(H, tile, stride)
    xs = window_starts(W, tile, stride)
    coords = [(y, x0) for y in ys for x0 in xs]
    tta_list = _tta_transforms(tta)
    n_tta = len(tta_list)
    prob_sum = torch.zeros((4, H, W), dtype=torch.float32, device=device)
    w_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    w_t = torch.from_numpy(hann2d(tile)).to(device)
    t0 = time.time()
    i = 0
    while i < len(coords):
        cur = min(batch, len(coords) - i)
        chunk = coords[i:i + cur]
        tiles = np.stack([x[y:y + tile, x0:x0 + tile].transpose(2, 0, 1)
                          for (y, x0) in chunk])
        xt = torch.from_numpy(tiles).to(device)
        try:
            acc = None
            for _name, fwd, inv in tta_list:
                probs = _probs(model, fwd(xt), amp_dtype)
                if probs is None:
                    raise NonFiniteLogits("logits no finitos en batch de eval")
                if model_b is not None:
                    probs_b = _probs(model_b, fwd(xt), amp_dtype)
                    if probs_b is None:
                        raise NonFiniteLogits("logits no finitos en batch de eval (B)")
                    probs = alpha * probs + (1.0 - alpha) * probs_b
                probs = inv(probs)
                acc = probs if acc is None else acc + probs
            acc = acc / n_tta
            for j, (y, x0) in enumerate(chunk):
                prob_sum[:, y:y + tile, x0:x0 + tile] += acc[j] * w_t
                w_sum[y:y + tile, x0:x0 + tile] += w_t
            i += cur
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch > 1:
                torch.cuda.empty_cache()
                batch = max(1, cur // 2)
                if verbose:
                    print(f"  [OOM] batch -> {batch}", flush=True)
            else:
                raise
    pred = (prob_sum / w_sum.clamp_min(1e-6)).argmax(dim=0)
    pred = pred.to(torch.uint8).cpu().numpy()
    if verbose:
        print(f"    inferencia: {len(coords)} ventanas en {time.time()-t0:.1f}s "
              f"({len(coords)/max(time.time()-t0, 1e-6):.1f} win/s)", flush=True)
    return pred


# ----------------------------------------------------------------------------
# metricas por pixel
# ----------------------------------------------------------------------------
def class_dice_iou(pred, gt, c):
    """Dice/IoU de la clase c. None si la clase no esta ni en pred ni en GT.

    Convencion: GT presente y pred ausente -> (0.0, 0.0).
    """
    p = pred == c
    g = gt == c
    ps, gs = int(p.sum()), int(g.sum())
    if ps + gs == 0:
        return None
    inter = int(np.logical_and(p, g).sum())
    dice = (2.0 * inter) / (ps + gs)
    union = ps + gs - inter
    iou = (inter / union) if union > 0 else 0.0
    return float(dice), float(iou)


def hd_pair(p, g, pad=32):
    """HD y HD95 (superficie) entre mascaras booleanas 2D. None si alguna vacia."""
    from scipy import ndimage as ndi
    p = p.astype(bool)
    g = g.astype(bool)
    if not p.any() or not g.any():
        return None
    ys, xs = np.where(p | g)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(p.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(p.shape[1], int(xs.max()) + pad + 1)
    pc, gc = p[y0:y1, x0:x1], g[y0:y1, x0:x1]
    st = ndi.generate_binary_structure(2, 1)
    ps_ = pc ^ ndi.binary_erosion(pc, structure=st)
    gs_ = gc ^ ndi.binary_erosion(gc, structure=st)
    if not ps_.any():
        ps_ = pc
    if not gs_.any():
        gs_ = gc
    dt_g = ndi.distance_transform_edt(~gs_)
    dt_p = ndi.distance_transform_edt(~ps_)
    d_pg = dt_g[ps_]
    d_gp = dt_p[gs_]
    hd = float(max(d_pg.max(), d_gp.max()))
    hd95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    return hd, hd95


def slice_dice(pred, gt, min_px=800):
    """Dice por componente conexa del GT (aprox. evaluacion por rodaja del paper)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(gt > 0)
    comps = []
    if n == 0:
        return comps, 0
    sizes = np.bincount(lab.ravel())
    for k in range(1, n + 1):
        if sizes[k] < min_px:
            continue
        m = lab == k
        rec = {}
        for c in FG_CLASSES:
            r = class_dice_iou(pred[m], gt[m], c)
            if r is not None:
                rec[c] = r[0]
        if rec:
            comps.append(rec)
    return comps, n


# ----------------------------------------------------------------------------
# evaluacion completa
# ----------------------------------------------------------------------------
def _agg_class_stats(per_image, c):
    vals = [im["dice"][c] for im in per_image.values() if c in im["dice"]]
    ious = [im["iou"][c] for im in per_image.values() if c in im["iou"]]
    return vals, ious


def evaluate_pair(pred, gt, want_slice=True):
    """Metricas de un par pred/gt uint8 (0..3) -> (rec, comps, n_comp).

    Mismas convenciones que la evaluacion completa: clase ausente en GT y pred
    -> se omite; presente en GT y no predicha -> dice 0; predicha sin GT -> no
    entra en la media (FP). rec incluye "inter" (interseccion en px) para poder
    agregar la metrica micro.
    """
    rec = {"dice": {}, "iou": {}, "hd": {}, "hd95": {}, "pred_px": {},
           "gt_px": {}, "inter": {}}
    for c in FG_CLASSES:
        r = class_dice_iou(pred, gt, c)
        p = pred == c
        g = gt == c
        ps, gs = int(p.sum()), int(g.sum())
        rec["pred_px"][c] = ps
        rec["gt_px"][c] = gs
        if r is None:
            continue
        rec["dice"][c] = r[0]
        rec["iou"][c] = r[1]
        rec["inter"][c] = int(np.logical_and(p, g).sum())
        if gs > 0:
            hd = hd_pair(p, g)
            if hd is not None:
                rec["hd"][c], rec["hd95"][c] = hd
    if want_slice:
        comps, n_comp = slice_dice(pred, gt)
    else:
        comps, n_comp = [], 0
    return rec, comps, n_comp


def aggregate_per_image(per_image):
    """Agrega {stem: rec} -> medias/micro (todo salvo rodajas/bad/elapsed)."""
    dice_lists = {c: [] for c in FG_CLASSES}
    iou_lists = {c: [] for c in FG_CLASSES}
    hd_lists = {c: [] for c in FG_CLASSES}
    hd95_lists = {c: [] for c in FG_CLASSES}
    micro = {c: [0, 0, 0] for c in FG_CLASSES}  # inter, pred_px, gt_px
    fp_only = {c: [] for c in FG_CLASSES}
    n_gt = {c: 0 for c in FG_CLASSES}
    n_pred = {c: 0 for c in FG_CLASSES}
    ratios_is_lv, ratios_is_aar = [], []
    for stem, rec in per_image.items():
        for c in FG_CLASSES:
            ps = rec["pred_px"][c]
            gs = rec["gt_px"][c]
            if gs > 0:
                n_gt[c] += 1
            if ps > 0:
                n_pred[c] += 1
            if c in rec["dice"]:
                if gs == 0:
                    fp_only[c].append(stem)
                else:
                    dice_lists[c].append(rec["dice"][c])
                    iou_lists[c].append(rec["iou"][c])
                    if c in rec["hd"]:
                        hd_lists[c].append(rec["hd"][c])
                        hd95_lists[c].append(rec["hd95"][c])
            micro[c][0] += rec.get("inter", {}).get(c, 0)
            micro[c][1] += ps
            micro[c][2] += gs
        if rec["gt_px"][1] > 0 and rec["gt_px"][3] > 0:
            ratios_is_lv.append((rec["gt_px"][3] / rec["gt_px"][1],
                                 rec["pred_px"][3] / rec["pred_px"][1]
                                 if rec["pred_px"][1] > 0 else 0.0))
        if rec["gt_px"][2] > 0 and rec["gt_px"][3] > 0:
            ratios_is_aar.append((rec["gt_px"][3] / rec["gt_px"][2],
                                  rec["pred_px"][3] / rec["pred_px"][2]
                                  if rec["pred_px"][2] > 0 else 0.0))

    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    def _std(xs):
        return float(np.std(xs)) if xs else None

    dice_mean = {c: _mean(dice_lists[c]) for c in FG_CLASSES}
    dice_std = {c: _std(dice_lists[c]) for c in FG_CLASSES}
    iou_mean = {c: _mean(iou_lists[c]) for c in FG_CLASSES}
    hd_mean = {c: _mean(hd_lists[c]) for c in FG_CLASSES}
    hd95_mean = {c: _mean(hd95_lists[c]) for c in FG_CLASSES}
    micro_dice = {}
    for c in FG_CLASSES:
        inter, ps, gs = micro[c]
        micro_dice[c] = (2.0 * inter / (ps + gs)) if (ps + gs) > 0 else None
    valid = [dice_mean[c] for c in FG_CLASSES if dice_mean[c] is not None]
    avg3 = float(np.mean(valid)) if valid else None
    valid_micro = [micro_dice[c] for c in FG_CLASSES if micro_dice[c] is not None]
    avg3_micro = float(np.mean(valid_micro)) if valid_micro else None
    hd95_valid = [hd95_mean[c] for c in FG_CLASSES if hd95_mean[c] is not None]
    hd_valid = [hd_mean[c] for c in FG_CLASSES if hd_mean[c] is not None]
    avg_hd95 = float(np.mean(hd95_valid)) if hd95_valid else None
    avg_hd = float(np.mean(hd_valid)) if hd_valid else None

    def ratio_stats(pairs_list):
        if not pairs_list:
            return None
        gts = np.array([p[0] for p in pairs_list], dtype=np.float64)
        prs = np.array([p[1] for p in pairs_list], dtype=np.float64)
        q = lambda a, p: float(np.percentile(a, p))
        return {
            "n": int(len(gts)),
            "gt_median": q(gts, 50), "gt_q25": q(gts, 25), "gt_q75": q(gts, 75),
            "pred_median": q(prs, 50), "pred_q25": q(prs, 25), "pred_q75": q(prs, 75),
            "mae": float(np.mean(np.abs(prs - gts))),
            "bias": float(np.mean(prs - gts)),
        }

    return {
        "dice_mean": dice_mean, "dice_std": dice_std,
        "dice_micro": micro_dice, "iou_mean": iou_mean,
        "hd_mean": hd_mean, "hd95_mean": hd95_mean,
        "avg3_mean": avg3, "avg3_micro": avg3_micro,
        "avg_hd": avg_hd, "avg_hd95": avg_hd95,
        "n_gt_images": n_gt, "n_pred_images": n_pred,
        "fp_only_images": fp_only,
        "ratio_is_lv": ratio_stats(ratios_is_lv),
        "ratio_is_aar": ratio_stats(ratios_is_aar),
    }


def evaluate_split(model, model_b, alpha, pairs, tile, stride, batch, device,
                   amp_dtype, tta, save_overlays=0, save_preds=1, outdir=None,
                   verbose=True):
    per_image = {}
    slice_acc = {c: [] for c in FG_CLASSES}
    n_slices_total = 0
    candidates = []
    bad_images = []
    t_start = time.time()

    for idx, (stem, img_path, mask_path) in enumerate(pairs, 1):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  !! no se pudo leer {img_path}, se omite", flush=True)
            bad_images.append(stem)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"  !! no se pudo leer {mask_path}, se omite", flush=True)
            bad_images.append(stem)
            continue
        gt = np.clip(np.rint(gt.astype(np.float32)), 0, 3).astype(np.uint8)

        t0 = time.time()
        try:
            pred = infer_full_image(model, model_b, alpha, img, tile=tile,
                                    stride=stride, batch=batch, device=device,
                                    amp_dtype=amp_dtype, tta=tta,
                                    verbose=verbose)
        except Exception as e:
            print(f"  !! error evaluando {stem}: {e}", flush=True)
            bad_images.append(stem)
            continue

        if save_preds and outdir:
            try:
                pdir = os.path.join(outdir, "preds")
                os.makedirs(pdir, exist_ok=True)
                cv2.imwrite(os.path.join(pdir, f"{stem}.png"), pred)
            except Exception as e:
                print(f"  !! no se pudo guardar la pred de {stem}: {e}", flush=True)

        rec, comps, _n_comp = evaluate_pair(pred, gt)
        n_slices_total += len(comps)
        for comp in comps:
            for c, d in comp.items():
                slice_acc[c].append(d)
        # overlay candidato (mas tejido primero)
        tissue = float((gt > 0).mean())
        if save_overlays > 0:
            candidates.append((tissue, stem, img, gt, pred))

        dmean = (float(np.mean(list(rec["dice"].values())))
                 if rec["dice"] else float("nan"))
        if verbose:
            print(f"  [{idx}/{len(pairs)}] {stem}: dice_media={dmean:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        per_image[stem] = rec

    agg = aggregate_per_image(per_image)

    # medias por rodaja (aprox. evaluacion por slice del paper)
    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    slice_mean = {c: _mean(slice_acc[c]) for c in FG_CLASSES}
    slice_valid = [slice_mean[c] for c in FG_CLASSES if slice_mean[c] is not None]
    slice_avg3 = float(np.mean(slice_valid)) if slice_valid else None

    # overlays reducidos
    overlay_dir = None
    if save_overlays > 0 and outdir and candidates:
        overlay_dir = os.path.join(outdir, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)
        for f in os.listdir(overlay_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(overlay_dir, f))
        candidates.sort(key=lambda t: -t[0])
        for k, (_tissue, stem, img, gt, pred) in enumerate(candidates[:save_overlays]):
            H, W = img.shape[:2]
            scale = min(1.0, 1400.0 / W)
            size = (max(1, int(W * scale)), max(1, int(H * scale)))
            small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            sm_gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
            sm_pred = cv2.resize(pred, size, interpolation=cv2.INTER_NEAREST)
            sep = np.full((size[1], 4, 3), 255, dtype=np.uint8)
            col_gt = CLASS_COLORS[np.clip(sm_gt, 0, 3)]
            col_pred = CLASS_COLORS[np.clip(sm_pred, 0, 3)]
            panel = np.concatenate([small, sep, col_pred, sep, col_gt], axis=1)
            cv2.imwrite(os.path.join(overlay_dir, f"overlay_{k:02d}_{stem}.png"),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    agg.update({
        "slice_dice_mean": slice_mean, "slice_avg3": slice_avg3,
        "n_slices": n_slices_total,
        "bad_images": bad_images,
        "elapsed_s": round(time.time() - t_start, 1),
    })
    return per_image, agg, overlay_dir


def find_pairs(root, split):
    img_dir = os.path.join(root, split, "img")
    mask_dir = os.path.join(root, split, "mask")
    exts = (".jpg", ".jpeg", ".png")
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)
            if f.lower().endswith(exts)}
    msks = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)
            if f.lower().endswith(exts)}
    stems = sorted(set(imgs) & set(msks))
    return [(s, os.path.join(img_dir, imgs[s]), os.path.join(mask_dir, msks[s]))
            for s in stems]


def print_summary(tag, agg, extra=""):
    print(f"\n=== {tag} {extra} ===")
    print(f"{'clase':<6}{'dice(media)':>13}{'n':>4}{'micro':>9}"
          f"{'iou(media)':>12}{'hd95':>9}{'hd':>9}")
    fv = lambda v, nd: (f"{v:.{nd}f}" if v is not None else "—")
    for c in FG_CLASSES:
        print(f"{CLS_NAMES[c]:<6}"
              f"{fv(agg['dice_mean'][c], 4):>13}"
              f"{agg['n_gt_images'][c]:>4}"
              f"{fv(agg['dice_micro'][c], 4):>9}"
              f"{fv(agg['iou_mean'][c], 4):>12}"
              f"{fv(agg['hd95_mean'][c], 2):>9}"
              f"{fv(agg['hd_mean'][c], 2):>9}")
    print(f"AVG3   {fv(agg['avg3_mean'], 4)} (micro {fv(agg['avg3_micro'], 4)}) | "
          f"HD95 medio {fv(agg['avg_hd95'], 2)} | HD medio {fv(agg['avg_hd'], 2)}")
    if agg["slice_avg3"] is not None:
        print(f"AVG3 por rodaja (aprox.): {agg['slice_avg3']:.4f} "
              f"sobre {agg['n_slices']} rodajas")
    for key, name in (("ratio_is_lv", "IS/LV"), ("ratio_is_aar", "IS/AAR")):
        r = agg[key]
        if r:
            print(f"ratio {name}: GT {r['gt_median']:.3f} "
                  f"[{r['gt_q25']:.3f}-{r['gt_q75']:.3f}] | "
                  f"pred {r['pred_median']:.3f} "
                  f"[{r['pred_q25']:.3f}-{r['pred_q75']:.3f}] | "
                  f"MAE {r['mae']:.3f} | sesgo {r['bias']:+.3f} (n={r['n']})")
    fps = {CLS_NAMES[c]: len(v) for c, v in agg["fp_only_images"].items() if v}
    if fps:
        print(f"imagenes con FP de clase sin GT: {fps}")
    if agg["bad_images"]:
        print(f"imagenes con error: {agg['bad_images']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cfg-b", default=None)
    ap.add_argument("--ckpt-b", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--amp", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--tta", default="none", choices=["none", "flips", "d4"])
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--save-overlays", type=int, default=0)
    ap.add_argument("--save-preds", type=int, default=1, choices=[0, 1])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    tile = args.tile or int(cfg.get("dataset", {}).get("tile_size", 512))
    stride = args.stride or max(tile // 2, 64)
    tag = args.tag or os.path.basename(args.outdir.rstrip("/"))
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(PROJECT_ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    amp_dtype = torch.bfloat16 if (args.amp == "bf16" and device == "cuda") else None

    ckpt_path = args.ckpt or cfg.get("ckpt_path", os.path.join(cfg["ckpt_dir"], "best.pt"))
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    model_b = None
    ckpt_b = None
    if args.cfg_b:
        cfg_b = load_cfg(args.cfg_b)
        model_b = build_model(cfg_b).to(device)
        ckpt_b_path = args.ckpt_b or cfg_b.get("ckpt_path",
                                              os.path.join(cfg_b["ckpt_dir"], "best.pt"))
        if not os.path.isabs(ckpt_b_path):
            ckpt_b_path = os.path.join(PROJECT_ROOT, ckpt_b_path)
        ckpt_b = torch.load(ckpt_b_path, map_location=device, weights_only=False)
        model_b.load_state_dict(ckpt_b["model"])
        model_b.eval()

    root = cfg["dataset"]["root"]
    pairs = find_pairs(root, args.split)
    if args.max_images > 0:
        pairs = pairs[:args.max_images]

    print(f"[{tag}] {len(pairs)} imagenes | tile {tile} stride {stride} "
          f"batch {args.batch} amp {args.amp} tta {args.tta} device {device}")
    print(f"[{tag}] ckpt {os.path.relpath(ckpt_path, PROJECT_ROOT)} "
          f"(epoch {ckpt.get('epoch')}, val_dice {ckpt.get('dice')})")
    if model_b is not None:
        print(f"[{tag}] ensemble alpha {args.alpha} | "
              f"{os.path.relpath(ckpt_b_path, PROJECT_ROOT)} "
              f"(epoch {ckpt_b.get('epoch')}, val_dice {ckpt_b.get('dice')})")

    per_image, agg, overlay_dir = evaluate_split(
        model, model_b, args.alpha, pairs, tile, stride, args.batch, device,
        amp_dtype, args.tta, save_overlays=args.save_overlays,
        save_preds=args.save_preds, outdir=outdir)

    result = {
        "tag": tag,
        "split": args.split,
        "cfg": args.cfg,
        "ckpt": os.path.relpath(ckpt_path, PROJECT_ROOT),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_dice": ckpt.get("dice"),
        "cfg_b": args.cfg_b,
        "ckpt_b": (os.path.relpath(ckpt_b_path, PROJECT_ROOT) if model_b is not None else None),
        "ckpt_b_epoch": (ckpt_b.get("epoch") if ckpt_b is not None else None),
        "alpha": (args.alpha if model_b is not None else None),
        "tile": tile, "stride": stride, "batch": args.batch,
        "amp": args.amp, "tta": args.tta, "save_preds": args.save_preds,
        "n_images": len(per_image),
        "images": per_image,
        "aggregate": agg,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_json = os.path.join(outdir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    print_summary(tag, agg, extra=f"(ep {ckpt.get('epoch')}, val {ckpt.get('dice')})")
    if overlay_dir:
        print(f"overlays -> {os.path.relpath(overlay_dir, PROJECT_ROOT)}/")
    print(f"metricas -> {os.path.relpath(out_json, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
