#!/usr/bin/env python
"""honest_summary.py — agrega las metricas honestas y las compara con el SOTA.

Lee runs/eval_honest/*/metrics.json (generados por src/eval_fullimage.py) y
escribe:
  - runs/eval_honest/SUMMARY.md
  - COMPARATIVA_HONESTA.md (copia en la raiz del proyecto)

Referencia SOTA (mismo dataset, split 70-15-15 por raton):
  DUCNet 2026 (Frontiers in Medicine) — AVG Dice 80.68 %
  (LV 84.22 / ARR 77.69 / IS 80.13), HD media 13.75 px.
"""
import json
import os

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(PROJECT, "runs", "eval_honest")

SINGLES = [
    ("A-v2", "U-Net EffB0 - 200 ep - dice_ce - bf16 - sched fix"),
    ("A-v3", "U-Net EffB0 - 100 ep - CE - sched fix"),
    ("A-v4", "U-Net EffB0 - 100 ep - CE - sched fix - aug v2"),
    ("G-v1", "Swin-T DPT - 100 ep"),
    ("G-v2", "Swin-T DPT - 200 ep - warmup - dice_ce - bf16"),
    ("G-v3", "Swin-T DPT - 300 ep - aug v2 - LLRD"),
    ("H", "PVTv2-B2 DPT - 100 ep"),
    ("I", "ConvNeXt-T DPT - 100 ep"),
    ("J", "ViT-B/16 DPT - 100 ep"),
]
DUCNET_AVG = 80.68
DUCNET_LV, DUCNET_AAR, DUCNET_IS, DUCNET_HD = 84.22, 77.69, 80.13, 13.75


def pct(v):
    return "—" if v is None else f"{v * 100:.2f}"


def num(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def load_runs():
    runs = {}
    if not os.path.isdir(OUT):
        return runs
    for d in sorted(os.listdir(OUT)):
        if d.startswith("smoke") or d.startswith("_"):
            continue
        p = os.path.join(OUT, d, "metrics.json")
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    runs[d] = json.load(f)
            except Exception as e:
                print(f"AVISO: no se pudo leer {p}: {e}")
    return runs


def table_row(tag, label, r):
    agg = r["aggregate"]
    d = agg["dice_mean"]
    lv, aar, isv = d.get("1"), d.get("2"), d.get("3")
    avg3 = agg["avg3_mean"]
    hd95 = agg["avg_hd95"]
    hd = agg["avg_hd"]
    rl = agg.get("ratio_is_lv") or {}
    mae = rl.get("mae")
    ep = r.get("ckpt_epoch")
    delta = f"{avg3 * 100 - DUCNET_AVG:+.2f}" if avg3 is not None else "—"
    return (f"| {tag} | {label} | {pct(avg3)} | {delta} | {pct(lv)} | {pct(aar)} "
            f"| {pct(isv)} | {num(hd95)} | {num(hd)} | {num(mae, 3)} | {ep} |")


def main():
    runs = load_runs()
    lines = []
    lines.append("# Comparativa HONESTA (full-image) — 16 oct 2026\n")
    lines.append(
        "Protocolo: inferencia de **ventana deslizante sobre la imagen completa** "
        "(6000x4000, solape 50%, mezcla Hann, argmax de probabilidad media). "
        "Dice por clase sobre mascara completa por imagen; media de imagenes con la "
        "clase presente en GT. HD/HD95 de superficie en px. "
        "Referencia SOTA: **DUCNet 2026, AVG 80.68 %** (mismo dataset y split).\n")
    lines.append("Generado por `src/honest_summary.py` a partir de "
                 "`runs/eval_honest/*/metrics.json`.\n")
    lines.append("| Run | Receta | AVG3 % | vs DUCNet | LV % | AAR % | IS % | HD95 px | HD px | MAE IS/LV | ep |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| **DUCNet 2026** | **referencia del paper** | **{DUCNET_AVG:.2f}** "
                 f"| — | {DUCNET_LV:.2f} | {DUCNET_AAR:.2f} | {DUCNET_IS:.2f} "
                 f"| — | {DUCNET_HD:.2f} | — | — |")

    used = set()

    def add_group(title, entries):
        found = [(t, l, runs[t]) for t, l in entries if t in runs]
        if not found:
            return
        lines.append("")
        lines.append(f"**{title}**")
        lines.append("")
        for t, l, r in found:
            lines.append(table_row(t, l, r))
            used.add(t)

    add_group("Modelos individuales", SINGLES)

    tta = [(f"{t}@tta", l + " - TTA flips") for t, l in SINGLES if f"{t}@tta" in runs]
    add_group("Con TTA (flips x4)", tta)

    ens = [(t, "ensemble probs alpha 0.5")
           for t in sorted(runs) if "+" in t]
    add_group("Ensembles (media de probabilidades)", ens)

    rest = [t for t in sorted(runs) if t not in used]
    other = [(t, "run adicional") for t in rest]
    add_group("Otros runs", other)

    lines.append("")
    lines.append("## Notas\n")
    lines.append("- `AVG3 %` = media de las tres clases de tejido (LV, AAR/ARR, IS), "
                 "el numero con el que se compara DUCNet (80.68).")
    lines.append("- `HD95`/`HD` en px; DUCNet reporta HD media 13.75 px "
                 "(nuestra columna `HD px` es la comparable directa).")
    lines.append("- La columna rodaja aproximada (`slice_avg3` en los JSON) "
                 "esta disponible en cada `metrics.json` (componentes conexas del GT).")
    lines.append("- Convencion Dice: clase ausente en GT y pred -> se omite; "
                 "presente en GT pero no predicha -> 0; predicha sin GT -> no entra "
                 "en la media (se cuenta como FP en los JSON).")
    lines.append("- Inferencia en bf16 (verificado == fp32 el 16-oct en U-Net y "
                 "Swin: diferencia < 0.0002 de Dice).")

    text = "\n".join(lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as f:
        f.write(text)
    with open(os.path.join(PROJECT, "COMPARATIVA_HONESTA.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
