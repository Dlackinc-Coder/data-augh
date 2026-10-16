"""Resumen de la noche: agrega resultados de eval/ensemble y escribe
COMPARATIVA_NOCHE.md con el estado de cada paso (para leer de un vistazo
por la manana)."""
import os
import re

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HIST = [
    ("A  U-Net+EffB0 (gamaliel, receta original)", 0.9400, 0.9264, "referencia"),
    ("G-v2 Swin-T DPT (200 ep, fix scheduler)", 0.9082, 0.8871, "mejor transformer"),
    ("G    Swin-T DPT (100 ep)", 0.9055, 0.8835, "historico"),
    ("B  SegFormer MiT-B2 v4", 0.9004, 0.8792, "historico"),
    ("H  DPT PVTv2-B2", 0.8979, 0.8764, "historico"),
    ("A-v2 U-Net (4 cambios + fix = 0.8969)", 0.8969, 0.8748, "diagnostico"),
    ("I  DPT ConvNeXt-T", 0.8895, 0.8665, "historico"),
    ("J  DPT ViT-B/16", 0.8509, 0.8275, "historico"),
]

EVALS = [
    ("A-v3 U-Net (solo fix scheduler)", "runs/eval_a_v3.log"),
    ("G-v3 Swin (aug2+LLRD+300ep)", "runs/eval_g_v3.log"),
    ("A-v4 U-Net (receta A + aug v2)", "runs/eval_a_v4.log"),
]

ENSS = [
    ("Ensemble A-v3 + G-v3", "runs/ens_a3_g3.log"),
    ("Ensemble A-v4 + G-v3", "runs/ens_a4_g3.log"),
    ("Ensemble A-v3 + G-v2", "runs/ens_a3_g2.log"),
    ("Ensemble A-v4 + G-v2", "runs/ens_a4_g2.log"),
]

TRAINS = [
    ("A-v3", "runs/train_a_v3.log"),
    ("G-v3", "runs/train_g_v3.log"),
    ("A-v4", "runs/train_a_v4.log"),
]


def read(path):
    full = os.path.join(PROJECT, path)
    if not os.path.exists(full):
        return ""
    with open(full) as f:
        return f.read()


def parse_test(txt):
    m = re.search(r"TEST \| dice ([0-9.]+) \| iou ([0-9.]+)", txt)
    return (float(m.group(1)), float(m.group(2))) if m else None


def parse_train(txt):
    best, last, n = 0.0, None, 0
    for m in re.finditer(r"epoch (\d+) \|.*val_dice ([0-9.]+)", txt):
        n += 1
        last = (int(m.group(1)), float(m.group(2)))
        best = max(best, float(m.group(2)))
    return n, best, last


def main():
    lines = []
    lines.append("# Resultados de la noche (15-16 oct 2026)\n")
    lines.append("Generado automaticamente por `src/night_summary.py`.\n")

    lines.append("## Estado de la cadena\n")
    status = read("runs/night_status.log")
    if status:
        lines.append("```")
        lines.append(status.strip())
        lines.append("```")
    else:
        lines.append("_Sin night_status.log — la cadena no arranco._")

    lines.append("\n## Entrenamientos\n")
    lines.append("| Run | Epocas | Best val | Ultima ep | Estado |")
    lines.append("|---|---|---|---|---|")
    for tag, path in TRAINS:
        txt = read(path)
        n, best, last = parse_train(txt)
        if n == 0:
            estado = "NO EJECUTADO" if not txt else "SIN EPOCAS"
        elif "done. best val dice" in txt:
            estado = "COMPLETADO"
        else:
            estado = "EN CURSO / INTERRUMPIDO"
        ult = f"{last[0]} ({last[1]:.4f})" if last else "—"
        lines.append(f"| {tag} | {n} | {best:.4f} | {ult} | {estado} |")

    lines.append("\n## Test (protocolo random, igual que toda la comparativa)\n")
    lines.append("| Modelo | Dice | IoU |")
    lines.append("|---|---|---|")
    for name, d, i, note in HIST:
        lines.append(f"| {name} | {d:.4f} | {f'{i:.4f}' if i else '—'} |")
    any_new = False
    for name, path in EVALS:
        r = parse_test(read(path))
        if r:
            any_new = True
            lines.append(f"| **{name}** | **{r[0]:.4f}** | **{r[1]:.4f}** |")
    if not any_new:
        lines.append("| _sin evals nuevos todavia_ | — | — |")

    lines.append("\n## Ensembles\n")
    lines.append("| Ensemble | Dice | IoU |")
    lines.append("|---|---|---|")
    any_ens = False
    for name, path in ENSS:
        r = parse_test(read(path))
        if r:
            any_ens = True
            lines.append(f"| **{name}** | **{r[0]:.4f}** | **{r[1]:.4f}** |")
    if not any_ens:
        lines.append("| _sin ensambles todavia_ | — | — |")

    lines.append("\n## Notas de contexto\n")
    lines.append("- Guided sampling: NO se uso a proposito — D/E/F rindieron "
                 "0.526 en test random (solo vieron tiles con tejido).")
    lines.append("- A-v3 cambia UNA sola variable vs A original (fix scheduler).")
    lines.append("- A-v4 = A + aug v2 (mide el efecto puro de la augmentacion).")
    lines.append("- G-v3 = Swin con aug v2 + LLRD(0.75) + 300 ep + warmup 10 + sin freeze.")
    lines.append("- Overlays: runs/eval_*/ y runs/ens_*/ (3 paneles + 1 panel).")

    out = os.path.join(PROJECT, "COMPARATIVA_NOCHE.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"resumen escrito: {out}")


if __name__ == "__main__":
    main()
