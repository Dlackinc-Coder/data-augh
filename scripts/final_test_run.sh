#!/bin/bash
# final_test_run.sh — aplica el protocolo final (elegido en VAL) UNA sola vez sobre TEST.
# Uso: final_test_run.sh raw|tta
#   raw = ensemble A-v4+G-v3 (media probs alpha=0.5)
#   tta = ensemble A-v4@tta+G-v3@tta (TTA flips x4)
# Protección one-shot: aborta si ya existe runs/eval_final/<tag>/metrics.json.
# Al final corre cleanup_eval (variantes "0,0," sin-op y "2,4000," limpieza) -> runs/eval_cleanup_final/.
cd "$HOME/projects/data-augh" || exit 1
PY="$HOME/miniconda3/envs/mioaug/bin/python"
MODE="${1:-raw}"

if [ "$MODE" = "tta" ]; then
  TAG="A-v4@tta+G-v3@tta"
  TTA=(--tta flips)
else
  TAG="A-v4+G-v3"
  TTA=()
fi

OUT="runs/eval_final/$TAG"
LOG="runs/eval_final_${TAG//[@+]/_}.log"
mkdir -p runs/eval_final runs/eval_cleanup_final

if [ -f "$OUT/metrics.json" ]; then
  echo "ABORTADO: ya existe $OUT/metrics.json (one-shot)"
  exit 1
fi

echo "== FINAL TEST [$MODE] -> $OUT ($(date '+%H:%M:%S'))"
{
"$PY" -u src/eval_fullimage.py \
  --cfg configs/seg_unet_a_v4.yaml --ckpt checkpoints/exp_unet_a_v4/best.pt \
  --cfg-b configs/seg_dpt_swin_v3.yaml --ckpt-b checkpoints/exp_dpt_swin_v3/best.pt \
  --alpha 0.5 "${TTA[@]}" --save-preds 1 --amp bf16 --batch 8 \
  --outdir "$OUT" --tag "$TAG"
echo "== LIMPIEZA OFFLINE sobre $OUT"
"$PY" -u src/cleanup_eval.py --run-dir "$OUT" \
  --grid "0,0," --grid "2,4000," \
  --out "runs/eval_cleanup_final/$TAG.json"
echo "== FINAL_TEST_DONE [$TAG] ($(date '+%H:%M:%S'))"
} 2>&1 | tee "$LOG"
