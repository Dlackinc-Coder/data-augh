# Protocolo DUCNet (paper) vs protocolo MioClinic (honesto)

Referencia: Wang et al., *DUCNet: dual-branch U-shaped network for myocardial
infarction segmentation*, Frontiers in Medicine 2026,
DOI 10.3389/fmed.2026.1836537. Dataset: Mouse Cardiac Infarction Image Dataset
(wwz4416, Apache-2.0) — el mismo que usamos.

## 1. Protocolo del paper (lo que hay que batir de forma comparable)

| punto | valor del paper |
|---|---|
| dataset | 243 ratones, 6000×4000, 4 clases (fondo/LV/AAR-ARR/IS) |
| split | 70/15/15 **por ratón** = 171/35/37 (idéntico al nuestro) |
| entrada | parches 256×256 |
| épocas | 60 |
| optimizador | AdamW, lr 1e-3, wd 0.05, momentum 0.9 |
| batch | 2 |
| métricas | Dice por clase (LV / ARR / IS) + media; HD; ratios de tamaño con IQR |
| resultado DUCNet | **AVG 80.68 %** (LV 84.22 / ARR 77.69 / IS 80.13), HD 13.75 |
| principales rivales (Tabla 1) | DSCNet 78.98 · EEMSNet 74.22 · LeViT-UNet 68.95 |

(Transcripción completa de la Tabla 1: informe v1.1 del 15-oct, sección 2.)

## 2. Protocolo histórico MioClinic (tiles aleatorios) — NO comparable

- Test: 8 tiles 512² aleatorios por imagen; Dice multiclase por lote de tiles.
- Problema medido el 15–16 oct: sobreestima la calidad real porque evalúa
  parches sueltos y no la segmentación completa del ratón.
  A-v2: 0.8969 (tiles) vs **0.7759** (honesto, full-image).
- Los números de A–F, G–J y variantes v2/v3 en `COMPARATIVA_*.md` usan este
  protocolo: válidos para ranking interno, **no** para citar contra el paper.

## 3. Protocolo honesto MioClinic (el que se reporta)

`src/eval_fullimage.py`:

- Ventana deslizante sobre la imagen completa 6000×4000 (tile 512, stride 256,
  mezcla Hann, argmax de la probabilidad media por píxel).
- Dice/IoU por imagen y clase con convención explícita:
  clase ausente en GT y pred → se omite; presente en GT y no predicha → 0;
  predicha sin estar en GT → no entra en la media, cuenta como FP y entra en
  la variante micro.
- HD y HD95 de superficie en px (por imagen/clase y global).
- Ratios IS/LV e IS/AAR: mediana + IQR (GT y pred), MAE y sesgo.
- Métrica secundaria “por rodaja” (componentes conexas del GT) para aproximar
  una evaluación por slice.
- TTA (flips ×4, d4) y ensembles de probabilidades (α configurable) con el
  mismo protocolo y métricas.

## 4. Diferencias que el informe debe declarar

1. **Granularidad de evaluación del paper**: el artículo no documenta al
   detalle cómo agrega el Dice de test (“all mask val test”). Asumimos máscara
   completa por ratón como comparación principal y reportamos también la
   variante por rodaja (más conservadora).
2. **HD**: el paper reporta un único valor medio (13.75 px). Nuestra HD
   honesta es sensible a islas lejanas sin post-proceso → reportamos HD95 y la
   distribución, además de la HD media.
3. **Receta de entrenamiento**: la nuestra (512², épocas propias, aug v1/v2)
   no replica exactamente la del paper; lo comparable es el *resultado* bajo
   el mismo split y protocolo de evaluación documentado.
4. **Selección de checkpoint**: usamos validación (35 ratones) como cualquier
   protocolo con val; el test (37) se toca una sola vez con la receta final
   (Fase 3). Históricamente hubo selección ruidosa por val-tiles; Fase 3 pasa
   a val full-image.

## 5. Criterio de victoria

- AVG3 honesto (media de LV/AAR/IS) **> 80.68** con el mismo split y protocolo
  documentado; idealmente también micro > 80.68.
- Secundarios de calidad: HD95 por debajo del rango del paper y ratios IS/LV,
  IS/AAR con MAE pequeño (cuantificación clínica).

## 6. Resultados (se actualiza al cerrar la Fase 2)

| run | AVG3 % | micro % | LV % | AAR % | IS % | HD95 px | fuente |
|---|---|---|---|---|---|---|---|
| DUCNet 2026 (paper) | 80.68 | — | 84.22 | 77.69 | 80.13 | — | DOI arriba |
| A-v2 (U-Net 200 ep) | 77.59 | 80.38 | 80.57 | 71.74 | 80.45 | 63.7 | `runs/eval_honest/A-v2` |
| A-v3 (U-Net 100 ep) | 72.70 | 76.11 | 77.06 | 69.28 | 71.77 | 180.2 | `runs/eval_honest/A-v3` |
| G-v1..J, G-v2/v3, A-v4, TTA, ensembles | — | — | — | — | — | — | en cola |
