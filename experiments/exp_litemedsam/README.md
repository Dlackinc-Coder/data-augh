# Experimento: LiteMedSAM (Modelo C - Medical Foundation Model)

> **Estado**: ✅ Completado | **Fecha**: Oct 2026 | **Resultado**: **Dice 85.21%** (test set)

---

## 📋 Configuración del Modelo

| Parámetro | Valor |
|-----------|-------|
| **Arquitectura** | TinyViT encoder (distilado de MedSAM/ViT) + decoder CNN propio |
| **Pre-entrenado** | LiteMedSAM (SAM → MedSAM 1.5M médicas → TinyViT distilado) |
| **Encoder** | TinyViT: embed_dims=[64,128,160,320], depths=[2,2,6,2], 5.73M params |
| **Decoder** | 3 bloques Conv-BN-GELU + classifier (0.39M params) |
| **Total params** | 6.11M |
| **Freeze encoder** | Épocas 1-15 (unfreeze en 16) |
| **Clases** | 4: 0=background, 1=LV, 2=AAR, 3=IS |
| **Input** | 512×512 → resize 256×256 (TinyViT) → output 512×512 |
| **Loss** | Dice + CE (class_weights [0.1,1,1,1]) |
| **Optimizador** | AdamW lr=3e-4, weight_decay=1e-4 |
| **Scheduler** | Cosine Annealing (100 épocas) |
| **Batch efectivo** | 8 (batch=4 + grad_accum=2) |
| **Precision** | AMP FP16 |
| **Hardware** | RTX 4060 8GB, CUDA 12.1 |
| **Tiempo** | ~99s/época, ~2.75h total |

---

## 🏆 Resultados (Test Set - 37 imágenes nunca vistas)

| Métrica | Valor | Comparación |
|---------|-------|-------------|
| **Dice** | **0.8521** | vs U-Net A: 0.9400 (-8.8pp) |
| **IoU** | **0.8144** | vs U-Net A: 0.9264 (-11.2pp) |
| **Best val dice** | 0.8442 (epoch 97) | |
| **Overlays** | 35 tiles con tejido ≥5% | |

---

## 📈 Curva de entrenamiento

| Época | train | val_dice | Nota |
|-------|-------|----------|------|
| 1 | 1.68 | 0.5921 | Inicio (encoder congelado) |
| 5 | 0.82 | 0.7861 | |
| 15 | 0.68 | 0.8061 | Última con encoder congelado |
| 16 | 0.68 | 0.7793 | **Unfreeze encoder** |
| 30 | 0.63 | 0.8311 | Mejora post-unfreeze |
| 50 | ~0.61 | ~0.84 | Estabilización |
| **97** | **0.576** | **0.8442** | **Best checkpoint** |
| 100 | 0.576 | 0.8242 | Final |

---

## 🔧 Notas de implementación

1. **Resize 512→256**: TinyViT soporta solo 256×256 (window attention fija). El wrapper resizea input a 256, procesa, y upsamplea output a 512.
2. **Encoder pre-entrenado médico**: 298 tensores cargados de `lite_medsam.pth` (0 missing).
3. **Dataset fix**: Entrenado con `dataset.py` corregido (masks 0-3 directos).
4. **Prompt-free**: LiteMedSAM original usa box prompts; aquí se usó solo el encoder TinyViT + decoder propio para segmentación semántica (comparable con A y B).

---

## 📁 Artefactos

- `checkpoints/best.pt` - Epoch 97 (val_dice 0.8442)
- `checkpoints/last.pt` - Epoch 100
- `logs/train_litemedsam.log` - Log completo
- `logs/events.out.tfevents.*` - TensorBoard
- `results/test_overlays/` - 35 overlays 3 paneles
- `results/test_overlays_1panel/` - 35 overlays 1 panel
- `config.yaml` - Configuración
