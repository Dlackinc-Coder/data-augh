# Comparativa HONESTA (full-image) — 16 oct 2026

Protocolo: inferencia de **ventana deslizante sobre la imagen completa** (6000x4000, solape 50%, mezcla Hann, argmax de probabilidad media). Dice por clase sobre mascara completa por imagen; media de imagenes con la clase presente en GT. HD/HD95 de superficie en px. Referencia SOTA: **DUCNet 2026, AVG 80.68 %** (mismo dataset y split).

Generado por `src/honest_summary.py` a partir de `runs/eval_honest/*/metrics.json`.

| Run | Receta | AVG3 % | vs DUCNet | LV % | AAR % | IS % | HD95 px | HD px | MAE IS/LV | ep |
|---|---|---|---|---|---|---|---|---|---|---|
| **DUCNet 2026** | **referencia del paper** | **80.68** | — | 84.22 | 77.69 | 80.13 | — | 13.75 | — | — |

**Modelos individuales**

| A-v2 | U-Net EffB0 - 200 ep - dice_ce - bf16 - sched fix | 77.59 | -3.09 | 80.57 | 71.74 | 80.45 | 63.74 | 446.00 | 0.467 | 88 |
| A-v3 | U-Net EffB0 - 100 ep - CE - sched fix | 72.70 | -7.98 | 77.06 | 69.28 | 71.77 | 180.20 | 294.38 | 0.616 | 20 |
| G-v1 | Swin-T DPT - 100 ep | 77.68 | -3.00 | 76.19 | 76.08 | 80.78 | 108.66 | 218.21 | 0.496 | 99 |
| G-v2 | Swin-T DPT - 200 ep - warmup - dice_ce - bf16 | 80.23 | -0.45 | 81.57 | 76.90 | 82.22 | 69.24 | 329.91 | 0.423 | 191 |
| H | PVTv2-B2 DPT - 100 ep | 77.55 | -3.13 | 78.66 | 74.27 | 79.72 | 64.78 | 282.29 | 0.442 | 87 |
| I | ConvNeXt-T DPT - 100 ep | 73.07 | -7.61 | 74.92 | 73.63 | 70.66 | 122.85 | 375.85 | 0.690 | 37 |
| J | ViT-B/16 DPT - 100 ep | 69.12 | -11.56 | 71.55 | 68.95 | 66.85 | 425.14 | 753.17 | 0.495 | 86 |

## Notas

- `AVG3 %` = media de las tres clases de tejido (LV, AAR/ARR, IS), el numero con el que se compara DUCNet (80.68).
- `HD95`/`HD` en px; DUCNet reporta HD media 13.75 px (nuestra columna `HD px` es la comparable directa).
- La columna rodaja aproximada (`slice_avg3` en los JSON) esta disponible en cada `metrics.json` (componentes conexas del GT).
- Convencion Dice: clase ausente en GT y pred -> se omite; presente en GT pero no predicha -> 0; predicha sin GT -> no entra en la media (se cuenta como FP en los JSON).
- Inferencia en bf16 (verificado == fp32 el 16-oct en U-Net y Swin: diferencia < 0.0002 de Dice).
