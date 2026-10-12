# Experimento: SegFormer MiT-B2 (Modelo B - Transformer)

> **Estado**: ⚠️ Completado con bug de datos | **Fecha**: Oct 2026 | **Resultado**: Dice 78.73% (INVÁLIDO - reentrenar)

---

## ⚠️ ADVERTENCIA: Bug de datos en este entrenamiento

**Este modelo fue entrenado con máscaras corruptas.** Durante la reescritura de `dataset.py`, la conversión de máscaras se cambió incorrectamente:

```python
# BUG (usado en este entrenamiento):
mask_tile = np.clip(np.rint(mask_tile / 255.0 * 3), 0, 3)  # TODO → 0

# CORREGIDO (original de Modelo A):
mask_tile = np.clip(np.rint(mask_tile.astype(np.float32)), 0, 3)  # correcto
```

**Efecto**: Todas las máscaras de entrenamiento se cargaron como clase 0 (background). El modelo aprendió a predecir solo background. Train loss → 0.0000, val_dice → 1.0000 (falso perfecto).

**Acción requerida**: Reentrenar con `dataset.py` corregido (ya aplicado).

---

## 📋 Configuración del Modelo

| Parámetro | Valor |
|-----------|-------|
| **Arquitectura** | SegFormer MiT-B2 (HuggingFace transformers) |
| **Pre-entrenado** | `nvidia/segformer-b2-finetuned-ade-512-512` (ImageNet-21k + ADE20K) |
| **Decoder** | Reinit 150 → 4 clases |
| **Freeze encoder** | Épocas 1-15 (unfreeze en 16) |
| **Clases** | 4: 0=background, 1=LV, 2=AAR, 3=IS |
| **Input tiles** | 512×512, 8 tiles/imagen/época |
| **Loss** | Dice + CE (class_weights [0.1,1,1,1]) |
| **Optimizador** | AdamW lr=3e-4, weight_decay=1e-4 |
| **Scheduler** | Cosine Annealing (100 épocas) |
| **Batch efectivo** | 8 (batch=2 + grad_accum=4) |
| **Precision** | AMP FP16 |
| **Hardware** | RTX 4060 8GB, CUDA 12.1 |

---

## 📊 Resultados (Test Set - CON BUG)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Dice** | 0.7873 | ⚠️ Modelo entrenado con masks rotas |
| **IoU** | 0.7837 | ⚠️ No comparable con Modelo A |
| **Época checkpoint** | 100 (last) | best.pt selection también afectado por bug |
| **Overlays** | 35 tiles con tejido | Guardados en results/ |

---

## 🔧 Issues conocidos

1. **Mask conversion bug**: `mask/255*3` → todo background (CORREGIDO en repo)
2. **val_dice siempre 1.0000**: Métrica por batch solo promedia clases presentes (conocido, eval.py lo corrige)
3. **best.pt = epoch 1**: Selección de best afectada por val_dice bugueado → se usó last.pt para eval
4. **eval.py get_logits**: No interpolaba output de SegFormer (128×128 → 512×512) - CORREGIDO

---

## 📁 Artefactos

- `checkpoints/best.pt` - Epoch 1 (inválido por bug val_dice)
- `checkpoints/last.pt` - Epoch 100 (evaluado → 0.7873)
- `logs/train.log` - Log completo de entrenamiento
- `logs/events.out.tfevents.*` - TensorBoard
- `results/test_overlays/` - 35 overlays 3 paneles
- `results/test_overlays_1panel/` - 35 overlays 1 panel
- `config.yaml` - Configuración del experimento
