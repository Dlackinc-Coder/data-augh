# Informe de la noche — 15→16 oct 2026

Ejecución autónoma (agente) de las Fases 0–2 del plan. Solo números reales.
La cadena nocturna original **no se tocó**: sigue corriendo en paralelo.

## 1. Hallazgo central: el protocolo de test histórico sobreestima

El test viejo (8 tiles aleatorios 512² por imagen, Dice por lote de tiles) mide
parches sueltos, no la segmentación completa del ratón. Medido hoy con el kit
nuevo sobre los mismos checkpoints:

| run | dice tiles (histórico) | dice honesto (full-image) |
|---|---|---|
| A-v2 (U-Net, ep 88) | 0.8969 | **0.7759** |
| A-v3 (U-Net, ep 20) | 0.8986 | **0.7270** |

→ La brecha es de ~12 puntos. Todos los números de `COMPARATIVA_*.md` sirven
para comparar modelos entre sí, pero **no** deben citarse contra el paper.

## 2. Kit de medición honesta (nuevo)

`src/eval_fullimage.py`: ventana deslizante sobre la imagen completa
(6000×4000, tile 512, stride 256, mezcla Hann, argmax de probabilidad media).

- Dice/IoU por imagen y clase (LV/AAR/IS), media sobre imágenes con la clase
  presente + variante micro global (los FP en imágenes sin la clase cuentan).
- HD y HD95 de superficie (px); ratios IS/LV e IS/AAR (mediana + IQR, MAE, sesgo).
- Métrica secundaria por rodaja (componentes conexas del GT).
- TTA (flips ×4 / d4) y ensembles de probabilidades integrados en el kit.
- Tests CPU: `tests/test_honest_metrics.py` (verdes) y `tests/test_metrics.py`
  para la métrica histórica de `src/eval.py` (reproduce el bug corregido el 9-oct).
- Verificado en GPU con la cadena entrenando: **bf16 == fp32** en U-Net
  (0.77995 vs 0.78017) y Swin (0.8659 vs 0.8660).

## 3. Resultados honestos (en curso, autónomo)

Primeros dos checkpoints terminados (referencia del paper en la primera fila):

| run | AVG3 % | micro % | LV % | AAR % | IS % | HD95 px | HD px | ratio IS/LV MAE |
|---|---|---|---|---|---|---|---|---|
| **DUCNet 2026** (paper, mismo split) | **80.68** | — | 84.22 | 77.69 | 80.13 | — (HD 13.75) | — | — |
| A-v2 (U-Net, 200 ep) | 77.59 | 80.38 | 80.57 | 71.74 | 80.45 | 63.7 | 446 | 0.467 |
| A-v3 (U-Net, 100 ep) | 72.70 | 76.11 | 77.06 | 69.28 | 71.77 | 180.2 | 294 | 0.616 |

La cola completa corre sola (`scripts/honest_eval_chain.sh`, desprendida):
G-v1 (en curso) → G-v2 → H → I → J → [espera fin de la cadena nocturna] →
G-v3, A-v4 → TTA flips (4 finalistas) → 4 ensembles → resumen agregado
(`runs/eval_honest/SUMMARY.md` + `COMPARATIVA_HONESTA.md`).

**Lecturas de los primeros datos:**
- A-v2 está a 3.1 pts de DUCNet en AVG3, pero a 0.3 pts en micro (80.38 vs
  80.68). El gap está concentrado en AAR (71.7 vs 77.7) y en errores de imagen.
- HD95/HD están dominados por islas lejanas de predicción (sin post-proceso):
  la limpieza de componentes en inferencia es la palanca más barata de la Fase 3.
- A-v3 (mejor val-tiles 0.8878) rinde honestamente peor que A-v2: la selección
  de checkpoint con val de tiles aleatorios es ruidosa → Fase 3: selección por
  val full-image (y/o por clase).

## 4. Fixes e infraestructura aplicados

- `src/night_summary.py`: H/I estaban intercambiados (confirmado contra la
  metadata de los checkpoints: H=PVTv2 val/época 87, I=ConvNeXt 37) + se
  añadieron G-v1 y J. Se ejecuta al final de la cadena.
- `src/train_seg.py`: **freeze inicial del encoder** (el freeze era no-op; solo
  existía el unfreeze). Neutral para la cadena actual (A-v4 no usa freeze).
  Aplicado con diff exacto + compile + import-test.
- TensorBoard: servía el logdir equivocado (clone viejo). Ahora `--logdir_spec`
  (proyectos + git), puerto 6006, HTTP 200.
- Git: rama `ubuntu-workspace` en `~/data-augh` (commit `449c36a` + este
  informe). **Sin push** — el repo público sigue pendiente de decisión.
- Backup: `~/backups/data-augh-2026-10-16_0127` (8.2 GB, sin dataset ni .venv)
  + bundle git 76 MB. Script: `scripts/backup_workspace.sh` (ya excluye .venv).

## 5. Estado de la cadena nocturna (intacta)

- G-v3: ep ~21 a las 01:30, val oscilando 0.84–0.90, ~68–76 s/época.
- Después: A-v4 (~1.5–2 h) → 4 ensembles → `night_summary` (con H/I corregido).
- Las evaluaciones honestas comparten GPU (~1 proceso de eval a la vez):
  la cadena terminará ~1 h más tarde de lo previsto (≈ mediodía).

## 6. Siguiente (Fase 3, cuando despierte)

1. Leer `runs/eval_honest/SUMMARY.md` completo (9 checkpoints + TTA + ensembles).
2. Matriz de recetas de 1-factor (val honesta): limpieza de componentes, loss
   para IS (Tversky), sampling, contexto 768/1024, aug v2/v3, selección de
   checkpoint por val full-image.
3. Top-2 recetas × 3 seeds → test congelado una sola vez → informe v2 contra
   DUCNet con el protocolo documentado (`docs/protocolo-duchnet-vs-mio.md`).

## 7. Dónde ver todo

- Resultados por run: `runs/eval_honest/<run>/metrics.json` (+ `overlays/`).
- Resumen agregado (al terminar): `runs/eval_honest/SUMMARY.md` y
  `COMPARATIVA_HONESTA.md` (raíz del proyecto).
- Log de la cola: `runs/honest_eval_chain.log`.
- Protocolo vs paper: `docs/protocolo-duchnet-vs-mio.md`.
- TensorBoard: `http://ubuntu:6006` (logdir_spec: proyectos + git).
