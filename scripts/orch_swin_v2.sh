#!/bin/bash
# Orquestador Swin v2: espera al entrenamiento (200 ep), corre eval fp32,
# guarda overlays y escribe resultados en COMPARATIVA_SWIN_V2.md
cd ~/projects/data-augh || exit 1
PY=~/miniconda3/envs/mioaug/bin/python
STATUS=runs/swin_v2_status.log
LOG=runs/train_dpt_swin_v2.log
EVALLOG=runs/eval_swin_v2.log

echo "[$(date)] orquestador swin v2 iniciado" >> "$STATUS"

# 1) esperar a que el train termine ("done") o morir
while true; do
  if grep -q "^done" "$LOG" 2>/dev/null; then
    echo "[$(date)] train OK: $(grep '^done' "$LOG")" >> "$STATUS"
    break
  fi
  if ! pgrep -f "train_seg.py configs/seg_dpt_swin_v2.yaml" > /dev/null; then
    echo "[$(date)] FALLO: proceso de train muerto sin 'done'" >> "$STATUS"
    tail -5 "$LOG" >> "$STATUS"
    exit 1
  fi
  sleep 60
done

# 2) eval fp32 sobre best.pt
echo "[$(date)] EVAL swin v2" >> "$STATUS"
mkdir -p experiments/exp_dpt_swin_v2
if ! $PY src/eval.py configs/seg_dpt_swin_v2.yaml --save-1panel > "$EVALLOG" 2>&1; then
  echo "[$(date)] FALLO eval (ver $EVALLOG)" >> "$STATUS"
  exit 1
fi
rm -rf experiments/exp_dpt_swin_v2/test_overlays experiments/exp_dpt_swin_v2/test_overlays_1panel
cp -r runs/test_overlays experiments/exp_dpt_swin_v2/test_overlays 2>/dev/null
cp -r runs/test_overlays_1panel experiments/exp_dpt_swin_v2/test_overlays_1panel 2>/dev/null

# 3) resultados en markdown (eval.py solo imprime "TEST | dice X | iou Y (ckpt epoch N)")
BEST=$(grep "^done" "$LOG" | sed 's/.*= //')
TESTR=$(grep "^TEST" "$EVALLOG" | tail -1)
TESTD=$(echo "$TESTR" | sed -n 's/.*dice \([0-9.]*\).*/\1/p')
TESTI=$(echo "$TESTR" | sed -n 's/.*iou \([0-9.]*\).*/\1/p')
BE=$(echo "$TESTR" | sed -n 's/.*ckpt epoch \([0-9]*\).*/\1/p')
{
  echo "# Swin v2 — scheduler corregido (warmup 5 + cosine completo) + 200 épocas"
  echo ""
  echo "- Fecha: $(date)"
  echo "- Config: configs/seg_dpt_swin_v2.yaml (seed 42, bf16, dice_ce, 512², lr 3e-4, freeze 15)"
  echo "- Fix: T_max ahora en optimizer steps; warmup 5 épocas; LR llega a 0 en ep 200"
  echo "- Best val dice: $BEST"
  echo "- Test: $TESTR"
  echo ""
  echo "## Comparación"
  echo "| Modelo | Test Dice | Test IoU |"
  echo "|---|---|---|"
  echo "| A — U-Net (base) | 0.9400 | 0.9264 |"
  echo "| G v1 — Swin 100 ep (scheduler viejo) | 0.9055 | 0.8835 |"
  echo "| **G v2 — Swin 200 ep (scheduler fix)** | **$TESTD** | **$TESTI** |"
} > COMPARATIVA_SWIN_V2.md

echo "[$(date)] COMPLETO: $TESTR" >> "$STATUS"
echo "[$(date)] SWIN V2 PIPELINE COMPLETADO" >> "$STATUS"
