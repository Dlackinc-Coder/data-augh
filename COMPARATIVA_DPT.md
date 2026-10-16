# Comparativa — Modelos DPT (transformers nuevos) vs baseline A–F

**Fecha:** 15 oct 2026 · **GPU:** RTX 4070 Ti 12 GB · **Protocolo idéntico a A–F:** 100 épocas, tiles 512², loss dice_ce, lr 3e-4, batch 2 × grad_accum 4, seed 42, freeze_encoder_epochs 15 (comportamiento igual en todos), test RANDOM (35 tiles con tejido ≥5%).

## Resultados (test RANDOM, dice / IoU)

| ID | Modelo | Backbone | Params | Test dice | Test IoU | Best val dice | s/época |
|----|--------|----------|-------:|----------:|---------:|--------------:|--------:|
| A | U-Net (baseline) | EfficientNet-B0 | — | **0.9400** | 0.9264 | — | — |
| **G** | **Swin-T + DPT** | swin_tiny_patch4_window7_224 | 46.4M | **0.9055** | 0.8835 | 0.9011 (ep 99) | 62.9 |
| B | SegFormer (baseline) | MiT-B2 | — | 0.9004 | 0.8792 | — | — |
| **H** | **PVTv2-B2 + DPT** | pvtv2_b2 | 43.1M | 0.8979 | 0.8764 | 0.9037 (ep 87) | 64.3 |
| **I** | **ConvNeXt-T + DPT** | convnext_tiny | 46.7M | 0.8895 | 0.8665 | 0.8834 (ep 37) | 63.0 |
| **J** | **ViT-B/16 + DPT** | vit_base_patch16_224 | 122.4M | 0.8509 | 0.8275 | 0.8768 (ep 86) | 67.7 |
| C | LiteMedSAM (baseline) | TinyViT | — | 0.8521 | 0.8144 | — | — |

Ranking: **A > G > B > H > I > J > C**.

## Conclusiones

1. **Swin-T (G) supera a SegFormer-B (B)**, el mejor transformer del paper original: 0.9055 vs 0.9004 (+0.5 pts). Es el segundo mejor modelo de toda la comparativa, solo por debajo de U-Net+EfficientNet-B0 (0.9400).
2. **PVTv2-B2 (H) queda prácticamente empatado con SegFormer** (0.8979 vs 0.9004, −0.25 pts) siendo el más ligero (43.1M).
3. **ConvNeXt-T (I) —aunque es CNN, no transformer— rinde por encima de LiteMedSAM** (0.8895) y cerca de SegFormer; buena opción eficiente.
4. **ViT-B/16 puro (J) es el más débil de los nuevos** (0.8509), apenas igualado con LiteMedSAM (0.8521): el mecanismo DPT compensa parte de la debilidad del ViT plano, pero sufre con datasets pequeños (37 imágenes de train). Además es el más grande (122M) y el más lento.
5. Los 4 modelos son viables en 12 GB (batch 2, ~63–68 s/época, 100 épocas ≈ 1h50m c/u).

## Hallazgo técnico importante (arreglado)

`src/eval.py` evaluaba con **autocast fp16** (dtype por defecto). **Swin-ViT desbordan en fp16** → logits NaN/Inf en **todos** los batches del test, y las métricas oficiales nocturnas de Swin (0.7873/0.4237) y ViT (0.7873/0.7837) se calcularon sobre predicciones corruptas (el IoU 0.4237 con dice 0.7873 es imposible: exige IoU ≥ dice/(2−dice) = 0.649).

- `eval.py` parcheado: inferencia en **fp32 exacto** + descarte y aviso de batches no-finitos.
- Verificado: fp32 == bf16 == fp16-sin-NaN en los 4 modelos (pvtv2 y convnext no se ven afectados, sus números nocturnos ya eran correctos).
- Baselines A–C verificados: sin NaN en fp16 y sus pares dice/IoU son consistentes → números válidos.
- Overlays de Swin/ViT regenerados con predicciones válidas (35 tiles c/u en `experiments/exp_dpt_*/test_overlays`).

## Artefactos

- Configs: `configs/seg_dpt_{swin,vit,pvtv2,convnext}.yaml`
- Checkpoints: `checkpoints/exp_dpt_*/best.pt`
- Logs: `runs/train_dpt_*.log`
- Código: `src/seg_model.py` → `build_dpt()` + dispatch `model.name: dpt`
