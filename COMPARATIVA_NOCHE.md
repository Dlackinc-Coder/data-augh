# Resultados de la noche (15-16 oct 2026)

Generado automaticamente por `src/night_summary.py`.

## Estado de la cadena

```
[2026-10-15 23:19:55] NIGHT START (orch_night.sh)
[2026-10-15 23:19:55] START train_a_v3
[2026-10-16 01:03:03] OK train_a_v3
[2026-10-16 01:03:03] START eval_a_v3
[2026-10-16 01:03:20] OK eval_a_v3
[2026-10-16 01:03:20] START train_g_v3
```

## Entrenamientos

| Run | Epocas | Best val | Ultima ep | Estado |
|---|---|---|---|---|
| A-v3 | 100 | 0.8878 | 100 (0.8805) | COMPLETADO |
| G-v3 | 30 | 0.9048 | 30 (0.8621) | EN CURSO / INTERRUMPIDO |
| A-v4 | 0 | 0.0000 | — | NO EJECUTADO |

## Test (protocolo random, igual que toda la comparativa)

| Modelo | Dice | IoU |
|---|---|---|
| A  U-Net+EffB0 (gamaliel, receta original) | 0.9400 | 0.9264 |
| G-v2 Swin-T DPT (200 ep, fix scheduler) | 0.9082 | 0.8871 |
| G    Swin-T DPT (100 ep) | 0.9055 | 0.8835 |
| B  SegFormer MiT-B2 v4 | 0.9004 | 0.8792 |
| H  DPT PVTv2-B2 | 0.8979 | 0.8764 |
| A-v2 U-Net (4 cambios + fix = 0.8969) | 0.8969 | 0.8748 |
| I  DPT ConvNeXt-T | 0.8895 | 0.8665 |
| J  DPT ViT-B/16 | 0.8509 | 0.8275 |
| **A-v3 U-Net (solo fix scheduler)** | **0.8986** | **0.8769** |

## Ensembles

| Ensemble | Dice | IoU |
|---|---|---|
| _sin ensambles todavia_ | — | — |

## Notas de contexto

- Guided sampling: NO se uso a proposito — D/E/F rindieron 0.526 en test random (solo vieron tiles con tejido).
- A-v3 cambia UNA sola variable vs A original (fix scheduler).
- A-v4 = A + aug v2 (mide el efecto puro de la augmentacion).
- G-v3 = Swin con aug v2 + LLRD(0.75) + 300 ep + warmup 10 + sin freeze.
- Overlays: runs/eval_*/ y runs/ens_*/ (3 paneles + 1 panel).
