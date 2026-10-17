#!/usr/bin/env python
"""make_cases.py — figuras cualitativas del test: Original | GT | IA | IA sobre imagen.

Protocolo final (el que firma el 81.21): ensemble A-v4@tta + G-v3@tta +
limpieza cl2_min4000 (close_r=2, min_cluster_px=4000). Reusa el kit real:
cleanup_pred (src/cleanup.py) + evaluate_pair (src/eval_fullimage.py) para
los casos; el Dice agregado de las 37 se recalcula con la misma formula.

Casos elegidos por percentil de AVG3 (sin cherry-picking):
  242 = P85 · 230 = mediana · 219 = P15 · 236 = peor (por transparencia)

Uso (desde la raíz del proyecto):  python scripts/make_cases.py
Salida: docs/img/caso_01_p85.png ... caso_04_peor.png + sanity-check en stdout.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from cleanup import cleanup_pred            # noqa: E402
from eval_fullimage import evaluate_pair    # noqa: E402

RUN = "runs/eval_final/A-v4@tta+G-v3@tta"
PRED_DIR = os.path.join(RUN, "preds")
TEST = "data/mouse_cardiac/test"
IMGDIR = "docs/img"
CLOSE_R, MIN_CLUSTER_PX = 2, 4000
FG = [1, 2, 3]

COLORS = {1: (0, 200, 0), 2: (0, 120, 255), 3: (255, 0, 0)}  # LV verde, AAR azul, IS rojo
BG_MASK = (45, 45, 45)
DARK = (32, 33, 36)

CASES = [
    ("caso_01_p85", "242", "Percentil 85"),
    ("caso_02_mediana", "230", "Mediana (percentil 50)"),
    ("caso_03_p15", "219", "Percentil 15"),
    ("caso_04_peor", "236", "Peor del test — incluido por transparencia"),
]

_TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_TTF_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def F(size, bold=False):
    for p in ([_TTF_B, _TTF] if bold else [_TTF]):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_img(cid):
    return np.array(Image.open(os.path.join(TEST, "img", f"{cid}.jpg")), dtype=np.uint8)


def load_gt(cid):
    gt = np.array(Image.open(os.path.join(TEST, "mask", f"{cid}.jpg")), dtype=np.float32)
    return np.clip(np.rint(gt), 0, 3).astype(np.uint8)  # = eval_fullimage l.414


def load_pred(cid):
    return np.array(Image.open(os.path.join(PRED_DIR, f"{cid}.png")), dtype=np.uint8)


def dice_inline(pred, gt, c):
    """Dice de una clase (None si la clase no esta en GT). Misma convencion:
    GT presente y pred ausente -> 0."""
    g = gt == c
    gs = int(g.sum())
    if gs == 0:
        return None
    p = pred == c
    inter = int(np.logical_and(p, g).sum())
    return 2.0 * inter / (int(p.sum()) + gs)


def dice_avg_from(rec):
    vals = [rec["dice"][c] for c in FG if c in rec["dice"] and rec["gt_px"][c] > 0]
    return float(np.mean(vals)) if vals else float("nan")


def colorize(mask):
    out = np.empty(mask.shape + (3,), np.uint8)
    out[:] = BG_MASK
    for c, col in COLORS.items():
        out[mask == c] = col
    return out


def blend(img, mask, alpha=0.5):
    out = img.astype(np.float32)
    for c, col in COLORS.items():
        m = mask == c
        out[m] = (1 - alpha) * out[m] + alpha * np.array(col, np.float32)
    return out.astype(np.uint8)


def resize_panel(arr, w, photo):
    im = Image.fromarray(arr)
    h = int(round(im.height * w / im.width))
    return im.resize((w, h), Image.LANCZOS if photo else Image.NEAREST)


def robust_bbox(gt, pred_c):
    """bbox del componente conexo principal de (gt|pred) cerrado — evita que
    ruido/speckle disperso haga explotar el recorte."""
    U = (gt > 0) | (pred_c > 0)
    Uc = ndi.binary_closing(U, iterations=3)
    lab, n = ndi.label(Uc)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = lab == int(np.argmax(sizes))
    ys, xs = np.where(main)
    return ys.min(), ys.max(), xs.min(), xs.max()


def make_case(name, cid, label, img, gt, pred_c, rec):
    bb = robust_bbox(gt, pred_c)
    y0, y1, x0, x1 = bb
    my = max(150, int(0.12 * (y1 - y0)))
    mx = max(150, int(0.12 * (x1 - x0)))
    y0 = max(0, y0 - my)
    y1 = min(img.shape[0] - 1, y1 + my)
    x0 = max(0, x0 - mx)
    x1 = min(img.shape[1] - 1, x1 + mx)

    img_c = img[y0:y1 + 1, x0:x1 + 1]
    gt_c = gt[y0:y1 + 1, x0:x1 + 1]
    pr_c = pred_c[y0:y1 + 1, x0:x1 + 1]

    W, GAP, MARG, LBL_H, CAP_H = 500, 24, 30, 56, 60
    panels = [
        resize_panel(img_c, W, photo=True),
        resize_panel(colorize(gt_c), W, photo=False),
        resize_panel(colorize(pr_c), W, photo=False),
        resize_panel(blend(img_c, pr_c), W, photo=False),
    ]
    ph = panels[0].height
    Wtot = MARG * 2 + 4 * W + 3 * GAP
    Htot = MARG + LBL_H + ph + CAP_H

    fig = Image.new("RGB", (Wtot, Htot), "white")
    d = ImageDraw.Draw(fig)
    labels = ["Original", "Máscara GT (referencia)", "Predicción IA (final)",
              "IA sobre la imagen"]
    fl = F(34, True)
    for i, (pan, lab) in enumerate(zip(panels, labels)):
        x = MARG + i * (W + GAP)
        w = d.textlength(lab, font=fl)
        d.text((x + (W - w) / 2, MARG + 8), lab, font=fl, fill=DARK)
        fig.paste(pan, (x, MARG + LBL_H))

    d1 = rec["dice"].get(1, float("nan"))
    d2 = rec["dice"].get(2, float("nan"))
    d3 = rec["dice"].get(3, float("nan"))
    avg = dice_avg_from(rec)
    cap = (f"{label} · test id {cid} — AVG3 {100 * avg:.1f}  "
           f"(LV {100 * d1:.1f} · AAR {100 * d2:.1f} · IS {100 * d3:.1f})")
    fc = F(32, True)
    w = d.textlength(cap, font=fc)
    if w > Wtot - 2 * MARG:
        fc = F(26, True)
        w = d.textlength(cap, font=fc)
    d.text(((Wtot - w) / 2, MARG + LBL_H + ph + 12), cap, font=fc, fill=DARK)

    fig.save(os.path.join(IMGDIR, f"{name}.png"))
    print(f"ok {name}: AVG3 {100 * avg:.1f} / LV {100 * d1:.1f} / "
          f"AAR {100 * d2:.1f} / IS {100 * d3:.1f}  [recorte {x1 - x0}x{y1 - y0}]")


def main():
    os.makedirs(IMGDIR, exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(PRED_DIR) if f.endswith(".png"))
    assert len(stems) == 37, f"esperaba 37 preds, hay {len(stems)}"
    case_ids = {c[1] for c in CASES}

    agg = {c: [] for c in FG}
    agg_inline = {c: [] for c in FG}
    case_preds = {}
    for cid in stems:
        gt = load_gt(cid)
        pred_c = cleanup_pred(load_pred(cid), close_r=CLOSE_R,
                              min_cluster_px=MIN_CLUSTER_PX)
        for c in FG:
            d_in = dice_inline(pred_c, gt, c)
            if d_in is not None:
                agg_inline[c].append(d_in)
        if cid in case_ids:
            case_preds[cid] = pred_c

    means = {c: float(np.mean(agg_inline[c])) for c in FG}
    avg3 = float(np.mean([means[c] for c in FG]))
    print("\n== SANITY cleaned (cl2_min4000) sobre las 37 ==")
    print(f"  LV  {100 * means[1]:.2f}   (objetivo 84.83)")
    print(f"  AAR {100 * means[2]:.2f}   (objetivo 75.70)")
    print(f"  IS  {100 * means[3]:.2f}   (objetivo 83.11)")
    print(f"  AVG3 {100 * avg3:.2f}   (objetivo 81.21)")

    print("\n== casos (cross-check evaluate_pair vs inline) ==")
    for name, cid, label in CASES:
        gt = load_gt(cid)
        pred_c = case_preds[cid]
        rec, _, _ = evaluate_pair(pred_c, gt, want_slice=False)
        ok = all(
            abs(rec["dice"][c] - dice_inline(pred_c, gt, c)) < 1e-12
            for c in FG if c in rec["dice"] and dice_inline(pred_c, gt, c) is not None
        )
        print(f"  {cid}: evaluate_pair vs inline {'OK' if ok else 'MISMATCH'}")
        img = load_img(cid)
        make_case(name, cid, label, img, gt, pred_c, rec)

    print("\nfiguras listas en", IMGDIR)


if __name__ == "__main__":
    main()
