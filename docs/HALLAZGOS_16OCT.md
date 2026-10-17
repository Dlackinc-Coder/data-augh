# Hallazgos 16-oct-2026 — Evaluación honesta full-image vs DUCNet (MioClinic)

Este documento consolida los resultados de la **re-evaluación honesta** (protocolo
full-image) de los checkpoints de la noche 15-16 oct, la comparación con el SOTA
(DUCNet 2026, AVG Dice 80.68 %), la selección de protocolo en validación y el
resultado final del protocolo congelado aplicado una vez en test.

Regla de oro usada en todo el documento: **solo números reales del kit honesto**
(`src/eval_fullimage.py`); los números "tile random" del protocolo viejo solo se
citan como ranking interno.

---

## 1. Protocolo honesto (qué mide y por qué)

- **Inferencia full-image**: ventana deslizante 512×512, stride 256 (solape 50 %),
  mezcla Hann de las probabilidades, argmax de la probabilidad media por píxel,
  sobre la imagen completa 6000×4000. Esto elimina el sesgo del protocolo viejo
  (`eval.py`: 30 tiles aleatorios por imagen), que sobreestimaba sistemáticamente
  porque muestreaba ventanas con tejido y descartaba los errores en zonas vacías.
- **Métricas** por clase de tejido (LV / AAR / IS):
  - Dice e IoU por imagen (media de imágenes con la clase presente en GT).
  - Convención: clase ausente en GT y pred → se omite; presente en GT y no
    predicha → Dice = 0 (cuenta en la media); predicha sin GT → no entra en la
    media (se cuenta como FP y en la métrica micro).
  - HD y HD95 de superficie (px) por imagen y clase.
  - Ratio de tamaño de infarto IS/LV e IS/AAR (GT vs pred, MAE/sesgo).
  - Métrica aproximada por "rodaja" (componentes conexas del GT).
- **Precisión**: bf16 verificado == fp32 (deltas < 0.0002 de Dice).
- **Split**: mismo que el paper (por ratón): train 171 / val 35 / test 37.
- **Equivalencia del kit**: el refactor de `evaluate_pair`/`aggregate_per_image`
  fue verificado **bit a bit** (re-ejecución de A-v3 reproduce el `metrics.json`
  anterior exactamente).

## 2. Resultados honestos en TEST (9 checkpoints + TTA + ensembles)

Tabla completa en `runs/eval_honest/SUMMARY.md`. Resumen (AVG3 = media LV/AAR/IS):

| Run | AVG3 % | vs DUCNet | LV % | AAR % | IS % | HD95 px | HD px |
|---|---|---|---|---|---|---|---|
| **DUCNet 2026 (ref.)** | **80.68** | — | 84.22 | 77.69 | 80.13 | — | 13.75 |
| G-v2 (Swin-T DPT, 200 ep) | 80.23 | −0.45 | 81.57 | 76.90 | 82.22 | 69.2 | 329.9 |
| G-v3 (Swin-T DPT, 300 ep, aug v2, LLRD) | 80.23 | −0.45 | 82.20 | 75.84 | 82.66 | 77.2 | 335.3 |
| A-v2 (U-Net EffB0, 200 ep) | 77.59 | −3.09 | 80.57 | 71.74 | 80.45 | 63.7 | 446.0 |
| G-v1 | 77.68 | −3.00 | 76.19 | 76.08 | 80.78 | 108.7 | 218.2 |
| H (PVTv2-B2) | 77.55 | −3.13 | 78.66 | 74.27 | 79.72 | 64.8 | 282.3 |
| A-v4 (U-Net EffB0, 100 ep, aug v2) | 76.22 | −4.46 | 82.99 | 69.87 | 75.80 | 186.7 | 335.3 |
| I (ConvNeXt-T) | 73.07 | −7.61 | 74.92 | 73.63 | 70.66 | 122.9 | 375.9 |
| A-v3 (U-Net EffB0, 100 ep) | 72.70 | −7.98 | 77.06 | 69.28 | 71.77 | 180.2 | 294.4 |
| J (ViT-B/16) | 69.12 | −11.56 | 71.55 | 68.95 | 66.85 | 425.1 | 753.2 |
| G-v2 @TTA (flips ×4) | 80.47 | −0.21 | 81.79 | 77.18 | 82.44 | 71.7 | 302.9 |
| **A-v4+G-v3 (ensemble probs α=0.5)** | **81.15** | **+0.47** | 84.65 | 75.80 | 83.02 | 99.7 | 210.9 |
| A-v4+G-v2 (ensemble) | 81.02 | +0.34 | 83.50 | 77.02 | 82.55 | 80.5 | 193.2 |
| A-v3+G-v3 (ensemble) | 81.02 | +0.34 | 85.21 | 75.35 | 82.51 | 121.2 | 232.9 |
| A-v3+G-v2 (ensemble) | 80.89 | +0.21 | 83.63 | 76.73 | 82.33 | 98.8 | 211.9 |

**Lectura**:
- **Los 4 ensembles U-Net+Swin superan a DUCNet en AVG3** (80.89–81.15 vs 80.68).
- El mejor individual queda a 0.21 del SOTA (G-v2@TTA 80.47).
- La ganancia del ensemble es real y consistente (viene de diversidad CNN vs
  transformer; +0.7-0.9 sobre el mejor miembro).
- Por clase: nuestro mejor ensemble supera el LV del paper (84.65 vs 84.22) y el
  IS (83.02 vs 80.13); el AAR queda algo por debajo (75.80 vs 77.69).

## 3. Selección de protocolo en VAL y resultado final

**Regla de decisión pre-registrada** (fijada antes de ver los números de val):
elegir por AVG3 (Dice) en val con la limpieza estándar aplicada, desempatando por
HD95 más bajo; el protocolo ganador se aplica **una sola vez** en test; TTA se
adopta solo si mejora de forma consistente en val (todos los singles).

### 3.1 Individuales en val (sin limpieza)

| Run | AVG3 % | LV % | AAR % | IS % | HD95 px | HD px |
|---|---|---|---|---|---|---|
| G-v3 | 82.75 | 85.00 | 82.84 | 80.42 | 78.3 | 235.6 |
| G-v2 | 81.08 | 83.59 | 81.84 | 77.80 | 89.5 | 248.4 |
| A-v4 | 73.42 | 84.24 | 76.15 | 59.87 | 531.7 | 746.5 |
| A-v3 | 72.47 | 78.73 | 76.85 | 61.84 | 174.2 | 325.4 |

(Nota: A-v4 en val muestra el problema de islas en su forma más extrema,
HD 746 → la limpieza por componentes es imprescindible para las CNN.)

### 3.2 TTA (flips ×4) en val

| Run | AVG3 % | Δ vs raw | HD95 px |
|---|---|---|---|
| G-v3@tta | 83.01 | +0.26 | 79.2 |
| G-v2@tta | 81.54 | +0.46 | 91.8 |
| A-v3@tta | 73.21 | +0.74 | 194.4 |
| A-v4@tta | 74.20 | +0.78 | 435.5 |

TTA mejora a **todos** los singles (+0.26 a +0.78) → se adopta como parte del
protocolo.

### 3.3 Ensembles y limpieza en val (decisión final)

Valores con limpieza estándar (min 4000 px; plateau 4000-8000 idéntico):

| Candidato | AVG3 % | HD95 px | HD px |
|---|---|---|---|
| **A-v4@tta+G-v3@tta ← GANADOR** | **83.40** | 101.7 | 228.2 |
| A-v4+G-v3 (ensemble raw) | 83.22 | 100.3 | 226.1 |
| G-v3@tta (single) | 83.01 | 79.2 | 201.7 |
| A-v3+G-v3 (ensemble) | 82.89 | 113.7 | 240.7 |
| G-v3 (single, limpio) | 82.76 | 78.3 | 201.9 |
| A-v4+G-v2 (ensemble) | 81.85 | 95.0 | 220.6 |
| A-v3+G-v2 (ensemble) | 81.64 | 101.4 | 225.1 |
| G-v2@tta (single) | 81.58 | 78.5 | 202.1 |
| G-v2 (single, limpio) | 81.11 | 80.6 | 203.7 |

Desglose por clase de los tres finalistas (val, limpiado min 4000):

| Run | LV % | AAR % | IS % | micro % |
|---|---|---|---|---|
| **A-v4@tta+G-v3@tta** | 87.83 | 83.01 | 79.36 | 85.03 |
| A-v4+G-v3 | 87.52 | 82.87 | 79.26 | 84.85 |
| G-v3@tta | 85.21 | 83.05 | 80.78 | 84.43 |

- TTA mejoró a los 4 singles (+0.26 a +0.78) **y también al ensemble** (+0.18:
  83.22 → 83.40) → se confirma como parte del protocolo.
- La limpieza apenas mueve el Dice de los ensembles (±0.01) pero sí arregla el HD
  de los que tenían islas claras (G-v2@tta: HD95 −13.3; A-v4@tta: HD95 −171 y
  −211 con 8000). En el ganador la limpieza es neutra en Dice (83.40 raw = limpio;
  HD95 101.4 → 101.7, diferencia ~ruido).
- Nota técnica: en algunos runs la limpieza sube el HD ~1 px al eliminar islas de
  true positive realmente pequeñas (<4000 px); el efecto en Dice es nulo.
- **Aplicada la regla pre-registrada (AVG3), el protocolo final es:** ensemble
  probs α=0.5 **A-v4 + G-v3** con **TTA flips ×4** + limpieza offline (close_r=2,
  min_cluster_px=4000). Val: **83.40 %**.

### 3.4 Resultado final en TEST (protocolo congelado, aplicado el sáb 17-oct)

Un único run **one-shot** (protegido contra repetición por el script) ejecutado
tras cerrar la selección en val:

```bash
cd ~/projects/data-augh && bash scripts/final_test_run.sh tta    # ~13 min GPU
```

| Test (protocolo final) | AVG3 % | vs DUCNet | LV % | AAR % | IS % | micro % | HD95 px | HD px |
|---|---|---|---|---|---|---|---|---|
| raw (control `0,0,`) | 81.22 | +0.54 | 84.83 | 75.70 | 83.13 | 83.80 | 100.8 | 211.8 |
| **+ limpieza (FINAL)** | **81.21** | **+0.53** | 84.83 | 75.70 | 83.11 | 83.79 | 101.4 | 212.4 |
| DUCNet 2026 (ref.) | 80.68 | — | 84.22 | 77.69 | 80.13 | — | — | 13.75 |

- Control de calidad ✅: la variante no-op (`0,0,`) reproduce el raw exactamente
  (81.22), verificado sobre la propia corrida final.
- **Lectura**: el protocolo final supera a DUCNet por **+0.53** puntos de AVG3
  (81.21 vs 80.68). Por clase: LV +0.61 e IS **+3.00**; el AAR queda por debajo
  (−1.99) — el margen limpio está en infarto y LV.
- TTA sobre el ensemble aportó +0.07 en test (81.15 → 81.22; en val había sido
  +0.18) y la limpieza resultó neutra en este ensemble (Δ Dice −0.01; Δ HD95
  +0.6 px) — coherente con que ya tenía muy pocas islas: se mantiene en el
  protocolo por ser inofensiva aquí y protectora en los modelos con islas.
- AVG3 por rodaja (aprox.): 74.74 sobre 185 rodajas.
- Métricas de ratio de volumen en test (secundarias, más ruidosas que en val):
  IS/LV MAE 0.474 (val 0.084); el modelo sobreestima IS/AAR (sesgo +1.36).
- Referencia del mismo framework sin TTA ni limpieza (medida el viernes):
  ensemble A-v4+G-v3 raw en test = 81.15 (+0.47).

## 4. Limpieza por componentes (post-proceso offline)

Las predicciones full-image contienen **islas lejanas** (falsos positivos en
zonas vacías) que inflan HD/HD95. Se implementó una poda por componentes
(`src/cleanup.py`): cierre morfológico r=2 del primer plano, etiquetado de
componentes, eliminación de las menores que `min_cluster_px`.

- Calibración en val con `src/cleanup_eval.py` (re-evaluación offline de preds
  guardadas; la variante "sin limpieza" reproduce `metrics.json` exactamente):
  - G-v2 val: min 4000 px → **+0.03 Dice, HD95 −8.9, HD −44.7**.
  - G-v3 val: min 4000 px → **+0.01 Dice, HD −33.7**.
  - Plateau entre 4000 y 20000 px: las islas del modelo están en el rango
    [2-4 k]px y los cortes reales son ≫ 20 k px → margen amplio y seguro.
- La limpieza **nunca quita todo** (si ningún cluster llega al umbral, conserva el mayor).
- Se aplica en evaluación offline (no en entrenamiento); 11 tests unitarios verdes.

## 5. Caveats de comparabilidad (honestidad metodológica)

1. **HD vs 13.75 del paper**: nuestro HD/HD95 es full-image (max sobre ~6 k px de
   espacio); el paper no especifica granularidad. Análisis propios:
   - El *speckle* de anotación del GT **no** es el ancla: quitar del GT los
     componentes < 800 px cambia el HD solo ~7 px máx (LV) y ~0 (IS) — el GT
     "ruidoso" está pegado a las regiones principales (≤7 px).
   - El HD residual lo domina **IS** (bandas finas): hd95 mediana en val ≈ 75 px,
     con casos de 742 px cuando el modelo falla la mitad de una banda de infarto.
   - Incluso midiendo **por rodaja** (bbox de cada corte): HD95 medio LV 33 /
     AAR 66 / IS 91 px. El 13.75 del paper no es reproducible con esta máscara
     GT y cualquier modelo a este Dice → la columna HD no es comparable 1:1.
   - La **métrica principal del paper (Dice AVG) sí es comparable** (mismo split).
2. **Valores 4-6 en las máscaras GT** (~0.1 % de píxeles; p.ej. test: 101 k px de
   4, 2.3 k de 5, 10 de 6): ruido de almacenamiento; el pipeline de entrenamiento
   también recorta a 0-3 (`dataset.py`), así que ambos lados son consistentes.
3. **Selección**: el protocolo final (modelo/ensemble, TTA, limpieza) se elige en
   **validación** y se aplica **una sola vez** en test (ver sección 3).

## 6. Reproducibilidad (comandos)

```bash
# Evaluación honesta de un checkpoint (guarda preds y métricas)
python -u src/eval_fullimage.py --cfg configs/seg_dpt_swin_v2.yaml \
  --ckpt checkpoints/exp_dpt_swin_v2/best.pt \
  --outdir runs/eval_honest/G-v2 --tag G-v2 --amp bf16 --batch 8

# Ensemble (media de probabilidades)
python -u src/eval_fullimage.py --cfg configs/seg_unet_a_v4.yaml \
  --ckpt checkpoints/exp_unet_a_v4/best.pt \
  --cfg-b configs/seg_dpt_swin_v3.yaml \
  --ckpt-b checkpoints/exp_dpt_swin_v3/best.pt --alpha 0.5 \
  --outdir runs/eval_honest/A-v4+G-v3 --tag A-v4+G-v3 --amp bf16 --batch 8

# Re-evaluación offline con limpieza por componentes
python -u src/cleanup_eval.py --run-dir runs/eval_honest_val/G-v2 \
  --grid "0,0," --grid "2,4000," --out runs/eval_cleanup_val/G-v2.json

# Cadenas completas (desprendidas, idempotentes)
bash scripts/honest_eval_chain.sh    # test: 9 ckpt + TTA + ensembles + SUMMARY
bash scripts/protocol_val_chain.sh   # val: finalistas + TTA + ensembles

# Protocolo final en test (one-shot protegido; evalúa + limpia automáticamente)
bash scripts/final_test_run.sh tta
```

## 7. Trabajo siguiente (Fase 3 ya preparada)

- `train_seg.py`: pérdida **Tversky** (`tversky_ce`, α=0.3 β=0.7 por defecto,
  acepta α/β por clase) + `save_every` para selección de checkpoints por val
  honesta; 6 tests unitarios verdes.
- Configs listas para entrenar (1-factor vs receta base):
  - `seg_unet_a_v5_tver` (A-v2 + Tversky), `seg_dpt_swin_v4_tver` (G-v3 + Tversky),
  - `seg_dpt_swin_v5_t12` (G-v3 + 12 tiles/epoch), `seg_dpt_swin_v6_ctx896`
    (G-v3 + contexto 896²).
- Prioridad sugerida: Tversky (recall de IS) → t12 → ctx896; mejores 2 recetas a
  3 seeds y evaluación final congelada una sola vez.
