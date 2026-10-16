# Swin v2 — scheduler corregido (warmup 5 + cosine completo) + 200 épocas

- Fecha: jue 15 oct 2026 15:55:07 CST
- Config: configs/seg_dpt_swin_v2.yaml (seed 42, bf16, dice_ce, 512², lr 3e-4, freeze 15)
- Fix: T_max ahora en optimizer steps; warmup 5 épocas; LR llega a 0 en ep 200
- Best val dice: 0.9104
- Test: TEST | dice 0.9082 | iou 0.8871 (ckpt epoch 191)

## Comparación
| Modelo | Test Dice | Test IoU |
|---|---|---|
| A — U-Net (base) | 0.9400 | 0.9264 |
| G v1 — Swin 100 ep (scheduler viejo) | 0.9055 | 0.8835 |
| **G v2 — Swin 200 ep (scheduler fix)** | **0.9082** | **0.8871** |
