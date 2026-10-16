# Informe de la noche — 15→16 oct 2026

Ejecución autónoma (agente) de las Fases 0–2 del plan. Solo números reales.
La cadena nocturna original **no se tocó**: sigue corriendo en paralelo.

## 1. Hallazgo central: el protocolo de test histórico sobreestima

El test viejo (8 tiles aleatorios 512² por imagen, Dice por lote de tiles) mide
parches sueltos, no la segmentación completa del ratón. Medido con el kit nuevo
sobre los mismos checkpoints, la brecha es de **~12 puntos** (A-v2: 0.8969 →
0.7759). Los números de `COMPARATIVA_*.md` sirven para ranking interno, pero
**no** deben citarse contra el paper.

## 2. Kit de medición honesta (nuevo)

`src/eval_fullimage.py`: ventana deslizante sobre la imagen completa
(6000×4000, tile 512, stride 256, mezcla Hann, argmax de probabilidad media).

- Dice/IoU por imagen y clase (LV/AAR/IS), media sobre imágenes con la clase
  presente + variante micro global (los FP en imágenes sin la clase cuentan).
- HD y HD95 de superficie (px); ratios IS/LV e IS/AAR (mediana + IQR, MAE, sesgo).
- Métrica secundaria por rodaja (componentes conexas del GT).
- TTA (flips ×4 / d4) y ensembles de probabilidades integrados en el kit.
- Tests CPU verdes: `tests/test_honest_metrics.py` y `tests/test_metrics.py`.
- Verificado en GPU con la cadena entrenando: **bf16 == fp32** en U-Net
  (0.77995 vs 0.78017) y Swin (0.8659 vs 0.8660).

## 3. Resultados honestos (7 de 9 ya evaluados; el resto, tras la cadena)

Tabla completa en `runs/eval_honest/SUMMARY.md` → copia `COMPARATIVA_HONESTA.md`.
(Se regenera al terminar la cola con G-v3, A-v4, TTA y ensembles.)

| run | AVG3 % | vs DUCNet | micro % | LV % | AAR % | IS % | HD95 px | ep |
|---|---|---|---|---|---|---|---|---|
| **DUCNet 2026** (paper) | **80.68** | — | — | 84.22 | 77.69 | 80.13 | — (HD 13.75) | — |
| **G-v2** (Swin-T, 200 ep) | **80.23** | **−0.45** | **82.54** | 81.57 | 76.90 | 82.22 | 69.2 | 191 |
| G-v1 (Swin-T, 100 ep) | 77.68 | −3.00 | 80.28 | 76.19 | 76.08 | 80.78 | 108.7 | 99 |
| A-v2 (U-Net, 200 ep) | 77.59 | −3.09 | 80.38 | 80.57 | 71.74 | 80.45 | 63.7 | 88 |
| H (PVTv2-B2, 100 ep) | 77.55 | −3.13 | 80.22 | 78.66 | 74.27 | 79.72 | 64.8 | 87 |
| I (ConvNeXt-T, 100 ep) | 73.07 | −7.61 | 76.70 | 74.92 | 73.63 | 70.66 | 122.9 | 37 |
| A-v3 (U-Net, 100 ep) | 72.70 | −7.98 | 76.11 | 77.06 | 69.28 | 71.77 | 180.2 | 20 |
| J (ViT-B/16, 100 ep) | 69.12 | −11.56 | 72.06 | 71.55 | 68.95 | 66.85 | 425.1 | 86 |

**Lecturas de los datos:**
- **G-v2 es el claro ganador de la fase 1**: a 0.45 pts de DUCNet en AVG3 y
  **por encima en micro** (82.54 vs 80.68), y todavía no lleva TTA, ni ensemble,
  ni limpieza de componentes (Fase 3) → cruzar 80.68 es muy plausible.
- G-v2 (Swin + 200 ep + warmup + dice_ce + bf16) es también
  el que mejor cuantifica el infarto (MAE IS/LV 0.42; el resto ≥ 0.44–0.69).
- Clúster 77.6: G-v1, A-v2 y H quedan a ~3 puntos; I/A-v3/J a 7–11 puntos.
- A-v3 (mejor val-tiles 0.8878) rinde honestamente peor que A-v2 → la selección
  de checkpoint con val-tiles es ruidosa (Fase 3: val full-image).
- HD/HD95 siguen dominados por islas lejanas (visibles en los overlays; J:
  HD95 425 px) → limpieza de componentes en inferencia: palanca #1 de Fase 3.
- Ranking tiles vs honesto apenas correlaciona (A-v3: 0.8986 tiles → 72.7
  honesto; G-v2: 0.9082 → 80.2): el protocolo viejo enmascaraba diferencias.

## 4. Fixes e infraestructura aplicados

- `src/night_summary.py`: H/I estaban intercambiados (confirmado contra la
  metadata de los checkpoints: H=PVTv2 ep 87, I=ConvNeXt ep 37) + G-v1 y J
  añadidos; validado ejecutándolo en real (recalcula bien, no explota).
- `src/train_seg.py`: **freeze inicial del encoder** (el freeze era no-op; solo
  existía el unfreeze). Neutral para la cadena actual (A-v4 no usa freeze).
  Aplicado con diff exacto + py_compile + import-test.
- TensorBoard: servía el logdir equivocado (clone viejo). Ahora `--logdir_spec`
  (proyectos + git), puerto 6006, HTTP 200.
- Git: rama `ubuntu-workspace` en `~/data-augh` (commits `449c36a`, `893cdaf`).
  **Sin push** — el repo público sigue pendiente de decisión.
- Backup: `~/backups/data-augh-2026-10-16_0127` (8.2 GB, sin dataset ni .venv)
  + bundle git 76 MB. Script: `scripts/backup_workspace.sh`.

## 5. Estado de la cadena nocturna (intacta)

- G-v3: ep ~40 a las 02:00, val 0.87–0.90 oscilando, ~77 s/época con la
  contención de las evals → ETA ~07:30–08:00.
- Después: A-v4 (~1.5–2 h) → 4 ensembles → `night_summary` → cadena completa
  ≈ 10:00. Luego la cola honesta evalúa G-v3 y A-v4 + TTA + ensembles →
  `SUMMARY.md`/`COMPARATIVA_HONESTA.md` finales ≈ 11:30–12:00.
- Ningún fallo en la cola honesta hasta ahora: 7/7 OK.

## 6. Siguiente (Fase 3, cuando despierte)

1. Leer `runs/eval_honest/SUMMARY.md` completo (G-v3, A-v4, TTA, ensembles).
2. Matriz de recetas de 1-factor (val honesta): limpieza de componentes, loss
   para IS (Tversky), sampling, contexto 768/1024, aug v2/v3, selección de
   checkpoint por val full-image.
3. Top-2 recetas × 3 seeds → test congelado una sola vez → informe v2 contra
   DUCNet con el protocolo documentado (`docs/protocolo-duchnet-vs-mio.md`).

## 7. Dónde ver todo

- Resultados por run: `runs/eval_honest/<run>/metrics.json` (+ `overlays/`).
- Tabla agregada: `runs/eval_honest/SUMMARY.md` y `COMPARATIVA_HONESTA.md`.
- Log de la cola: `runs/honest_eval_chain.log`.
- Protocolo vs paper: `docs/protocolo-duchnet-vs-mio.md`.
- TensorBoard: `http://ubuntu:6006` (logdir_spec: proyectos + git).
