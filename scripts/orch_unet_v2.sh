#!/bin/bash
# Orquestador U-Net A-v2: espera al entrenamiento (200 ep), corre eval fp32,
# guarda overlays y escribe resultados en COMPARATIVA_UNET_V2.md
cd ~/projects/data-augh || exit 1
PY=~/miniconda3/envs/mioaug/bin/python
STATUS=runs/unet_v2_status.log
LOG=runs/train_unet_v2.log
EVALLOG=runs/eval_unet_v2.log

echo "[$(date)] orquestador unet v2 iniciado" >> "$STATUS"

# 1) esperar a que el train termine ("done") o morir
while true; do
  if grep -q "^done" "$LOG" 2>/dev/null; then
    echo "[$(date)] train OK: $(grep '^done' "$LOG")" >> "$STATUS"
    break
  fi
  if ! pgrep -f "train_seg.py configs/seg_unet_effb0_v2.yaml" > /dev/null; then
    echo "[$(date)] FALLO: proceso de train muerto sin 'done'" >> "$STATUS"
    tail -5 "$LOG" >> "$STATUS"
    exit 1
  fi
  sleep 60
done

# 2) eval fp32 sobre best.pt
echo "[$(date)] EVAL unet v2" >> "$STATUS"
mkdir -p experiments/exp_unet_effb0_v2
if ! $PY src/eval.py configs/seg_unet_effb0_v2.yaml --save-1panel > "$EVALLOG" 2>&1; then
  echo "[$(date)] FALLO eval (ver $EVALLOG)" >> "$STATUS"
  exit 1
fi
rm -rf experiments/exp_unet_effb0_v2/test_overlays experiments/exp_unet_effb0_v2/test_overlays_1panel
cp -r runs/test_overlays experiments/exp_unet_effb0_v2/test_overlays 2>/dev/null
cp -r runs/test_overlays_1panel experiments/exp_unet_effb0_v2/test_overlays_1panel 2>/dev/null

# 3) resultados en markdown
BEST=$(grep "^done" "$LOG" | sed 's/.*= //')
TESTR=$(grep "^TEST" "$EVALLOG" | tail -1)
TESTD=$(echo "$TESTR" | sed -n 's/.*dice \([0-9.]*\).*/\1/p')
TESTI=$(echo "$TESTR" | sed -n 's/.*iou \([0-9.]*\).*/\1/p')
BE=$(echo "$TESTR" | sed -n 's/.*ckpt epoch \([0-9]*\).*/\1/p')
{
  echo "# U-Net A-v2 — scheduler corregido (warmup 5 + cosine completo) + 200 épocas"
  echo ""
  echo "- Fecha: $(date)"
  echo "- Config: configs/seg_unet_effb0_v2.yaml (seed 42, bf16, dice_ce, 512², lr 3e-4, freeze 15)"
  echo "- Fix: T_max en optimizer steps; warmup 5 épocas; LR llega a 0 en ep 200"
  echo "- Nota: A original (0.9400) fue 100 ep, crossentropy, fp16, scheduler viejo (gamaliel)"
  echo "- Best val dice: $BEST"
  echo "- Test: $TESTR"
  echo ""
  echo "## Comparación"
  echo "| Modelo | Test Dice | Test IoU |"
  echo "|---|---|---|"
  echo "| A — U-Net v1 (100 ep, CE, fp16, scheduler viejo) | 0.9400 | 0.9264 |"
  echo "| **A-v2 — U-Net (200 ep, dice_ce, bf16, scheduler fix)** | **$TESTD** | **$TESTI** |"
  echo "| G v2 — Swin 200 ep (scheduler fix) | 0.9082 | 0.8871 |"
} > COMPARATIVA_UNET_V2.md

echo "[$(date)] COMPLETO: $TESTR" >> "$STATUS"
echo "[$(date)] UNET V2 PIPELINE COMPLETADO" >> "$STATUS"
