# U-Net A-v2 — scheduler corregido (warmup 5 + cosine completo) + 200 épocas

- Fecha: jue 15 oct 2026 19:45:27 CST
- Config: configs/seg_unet_effb0_v2.yaml (seed 42, bf16, dice_ce, 512², lr 3e-4, freeze 15)
- Fix: T_max en optimizer steps; warmup 5 épocas; LR llega a 0 en ep 200
- Nota: A original (0.9400) fue 100 ep, crossentropy, fp16, scheduler viejo (gamaliel)
- Best val dice: 0.9085
- Test: TEST | dice 0.8969 | iou 0.8748 (ckpt epoch 88)

## Comparación
| Modelo | Test Dice | Test IoU |
|---|---|---|
| A — U-Net v1 (100 ep, CE, fp16, scheduler viejo) | 0.9400 | 0.9264 |
| **A-v2 — U-Net (200 ep, dice_ce, bf16, scheduler fix)** | **0.8969** | **0.8748** |
| G v2 — Swin 200 ep (scheduler fix) | 0.9082 | 0.8871 |
