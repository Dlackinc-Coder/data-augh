# Comparativa Final: 6 Modelos × 2 Test Sets

Todos los modelos entrenados 100 épocas, dataset `wwz4416/mouse-cardiac-dataset` (243 pares,
6000×4000, 4 clases: fondo, LV, AAR, IS). RTX 4060 8GB (512×512).

## Configuración experimental

| ID | Modelo | Sampling train | Tiles/época | Precision | Test dice |
|----|--------|----------------|-------------|-----------|-----------|
| A | U-Net + EfficientNet-B0 | Random | 8 | fp16 | 0.9400 / 0.9264 (random) |
| B | SegFormer MiT-B2 v4 | Random | 8 | bf16 | 0.9004 / 0.8792 (random) |
| C | LiteMedSAM (TinyViT) | Random | 8 | fp16 | 0.8521 / 0.8144 (random) |
| D | U-Net + EfficientNet-B0 | **Guided** | 4 | fp16 | 0.8486 / 0.7568 (guided) |
| E | SegFormer MiT-B2 | **Guided** | 4 | bf16 | 0.8456 / 0.7515 (guided) |
| F | LiteMedSAM (TinyViT) | **Guided** | 4 | fp16 | 0.8538 / 0.7606 (guided) |

## Test RANDOM (tiles aleatorios; 35 tiles con tejido ≥5%, tejido medio 3.5%)

| Modelo | Dice | IoU |
|--------|------|-----|
| **A** U-Net (random) | **0.9400** | **0.9264** |
| **B** SegFormer (random) | **0.9004** | **0.8792** |
| **C** LiteMedSAM (random) | **0.8521** | **0.8144** |
| D U-Net (guided) | 0.5260 | 0.4939 |
| E SegFormer (guided) | 0.5296 | 0.4926 |
| F LiteMedSAM (guided) | 0.4877 | 0.4354 |

> Los modelos guided (D/E/F) pierden ~0.4 Dice en este set: solo vieron tiles con tejido
> durante el entrenamiento (4 por imagen) y fallan en tiles de puro fondo.

## Test GUIDED (148 tiles con tejido ≥5%, tejido medio 45%)

| Modelo | Dice | IoU | Δ vs su baseline random |
|--------|------|-----|--------------------------|
| A U-Net (random) | 0.7923 | 0.6870 | — |
| B SegFormer (random) | 0.8089 | 0.7022 | — |
| C LiteMedSAM (random) | 0.8077 | 0.6999 | — |
| **D** U-Net (guided) | **0.8486** | **0.7568** | **+0.056** |
| **E** SegFormer (guided) | **0.8456** | **0.7515** | **+0.037** |
| **F** LiteMedSAM (guided) | **0.8538** | **0.7606** | **+0.046** |

## Conclusiones

1. **Guided sampling mejora la segmentación en tejido**: D/E/F superan a A/B/C en
   **+3.7% a +6.2% Dice** en el test guiado (las 3 arquitecturas, consistente).
2. **Trade-off**: el entrenamiento guiado (4 tiles/imagen, 100% tejido) degrada la
   robustez a fondo puro (test random ~0.5 vs ~0.9).
3. **Mejor en tejido**: F (LiteMedSAM guided), 0.8538 Dice / 0.7606 IoU.
4. **Mejor en tiles mixtos sin recorte previo**: A (U-Net random), 0.9400 Dice —
   parcialmente inflado por la abundancia de fondo fácil.
5. **bf16 fue decisivo en SegFormer** (B/E): elimina el NaN en BatchNorm que mataba
   los runs en fp16 tras unfreeze (epoch ~21).
6. **Recomendación producción**: si hay recorte previo al tejido → F (guided);
   si la imagen llega completa → A (random) o D con `exploration_rate` más alto.

## Métricas de entrenamiento

| Modelo | Best epoch (val) | Tiempo/época | VRAM pico |
|--------|------------------|--------------|-----------|
| A (512²) | 49 | ~365 s | ~1 GB |
| B v4 (512²) | 98 (val 0.9098) | ~93 s | ~3 GB |
| C (512²) | 97 (val 0.8442) | ~99 s | ~6 GB |
| D (512², 4 tiles) | 98 | ~43 s | ~1 GB |
| E (512², 4 tiles) | 72 | ~48 s | ~3 GB |
| F (512², 4 tiles) | 69 (val 0.8561) | ~51 s | ~6 GB |

## Artefactos

- Checkpoints: `checkpoints/exp_*/best.pt` (gitignorado — disponibles en el server)
- Overlays test: `experiments/exp_*/results/test_overlays{,_1panel}/`
- Logs: `experiments/exp_*/logs/`, `runs/`
- Pipeline automático: `scripts/pipeline_def.sh`
- Guided sampling: `src/dataset.py` (`_build_prob_maps`, `_guided_sample`)
