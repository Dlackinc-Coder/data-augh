# Experimento: U-Net + EfficientNet-B0 (Modelo A - Baseline)

> **Estado**: ✅ Completado | **Fecha**: Oct 2026 | **Resultado**: **Dice 94.00%** (supera paper DUCNet 80.68%)

---

## 📋 Configuración del Modelo

| Parámetro | Valor |
|-----------|-------|
| **Arquitectura** | U-Net (segmentation_models_pytorch) + EfficientNet-B0 encoder |
| **Encoder weights** | ImageNet (pre-trained) → fine-tuning completo |
| **Clases** | 4: 0=background, 1=LV (miocardio viable), 2=AAR (área riesgo), 3=IS (infarto) |
| **Input tiles** | 512×512, stride 512, 8 tiles/imagen/época |
| **Augmentación** | Albumentations: flip, rotate±15°, affine (scale 0.8-1.2), GaussNoise, brightness/contrast |
| **Loss** | CrossEntropyLoss (sin pesos de clase) |
| **Optimizador** | AdamW lr=3e-4, weight_decay=1e-4 |
| **Scheduler** | Cosine Annealing (100 épocas, 3e-4 → 0) |
| **Batch efectivo** | 8 (batch=2 + grad_accum=4) |
| **Precision** | AMP FP16 |
| **Muestreo tiles** | Aleatorio uniforme (sin ponderar por clase) |
| **Hardware** | RTX 4060 8GB, CUDA 12.1 |

---

## 📊 Dataset

| Aspecto | Detalle |
|---------|---------|
| **Fuente** | `wwz4416/mouse-cardiac-dataset` (Apache-2.0) |
| **Protocolo** | TTC + Evans Blue (ratón) |
| **Imágenes** | 243 pares, 6000×4000 JPEG RGB |
| **Máscaras** | 4 clases (0-3), 6000×4000 JPEG 1 canal |
| **Split** | 171 train / 35 val / 37 test (70/15/15 a nivel ratón) |

---

## 🏆 Resultados (Test Set - 37 imágenes nunca vistas)

| Métrica | Valor | Comparación |
|---------|-------|-------------|
| **Dice Promedio** | **0.9400** | 🏆 +13.3% vs DUCNet (80.68%) |
| **IoU Promedio** | **0.9264** | — |
| **Best Val Dice** | 0.9406 (epoch 49) | — |
| **Dice LV** | ~0.95+ | — |
| **Dice AAR** | ~0.92+ | — |
| **Dice IS** | ~0.90+ | 🏆 +10% vs DUCNet (80.13%) |
| **Tiempo total** | ~6.2 horas | 100 épocas × 365s |
| **VRAM pico** | ~1.0 GB / 8 GB | 87% libre |
| **Parámetros** | 6.3M | 38× menos que DUCNet |

---

## 📁 Artefactos Generados

```
experiments/exp_unet_efficientnet_b0/
├── config.yaml                 # Configuración completa
├── metrics.json                # Métricas en JSON
├── checkpoints/
│   ├── best.pt                 # Mejor modelo (epoch 49, val_dice 0.9406)
│   └── last.pt                 # Última época (epoch 100)
├── logs/
│   ├── train.log               # Log de entrenamiento (100 líneas)
│   ├── tb.log                  # Log TensorBoard
│   └── events.out.tfevents.*   # Eventos TB para visualización
└── results/
    ├── test_overlays/          # 64 overlays 3 paneles (orig | GT | pred)
    └── test_overlays_1panel/   # 64 overlays 1 panel (cian LV, amarillo AAR, rojo IS)
```

### Overlays - Filtrado Inteligente
- **Criterio**: `tissue_ratio ≥ 5%` (píxeles no-fondo en GT)
- **Total tiles test**: ~3,552 → **378 con tejido** → **Top 64 guardados** (ordenados por tissue_ratio desc)
- **Resultado**: 100% de overlays muestran tejido cardíaco (0 fondos vacíos)

---

## 🔍 Análisis y Conclusiones

### ✅ Fortalezas
1. **Rendimiento superior**: 94% Dice vs 80.68% paper DUCNet
2. **Eficiencia extrema**: 6.3M params, 1GB VRAM, 6 min/época
3. **Generalización**: Train/val losses paralelas, val_dice estable 0.92-0.94
4. **Listo producción**: Checkpoint `best.pt` validado en test hold-out

### ⚠️ Limitaciones conocidas
- Muestreo tiles aleatorio (no ponderado por clase)
- Loss CrossEntropy simple (sin Dice loss ni pesos de clase)
- Tiles 512² (contexto limitado vs 1024² en 12GB+)

### 🚀 Próximos pasos (Modelo B - Vision Transformer)
- Probar **SegFormer MiT-B2** (pre-trained ImageNet-21k) con transfer learning
- Loss Dice+CE combinada + freeze encoder 15 épocas
- Comparar: Dice IS, bordes irregulares, eficiencia

---

## 📈 Visualización TensorBoard

```bash
# En servidor 4060:
tensorboard --logdir runs --bind_all --port 6006
# Abrir en laptop: http://100.81.212.52:6006
```

Métricas loggeadas: `loss/train`, `loss/val`, `metric/val_dice`, `lr`

---

## 🏷️ Tags
`baseline` `unet` `efficientnet-b0` `mouse-cardiac` `ttc-evans-blue` `dice-94` `production-ready`
