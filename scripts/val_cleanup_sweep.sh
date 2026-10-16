#!/bin/bash
# val_cleanup_sweep.sh — rejilla de limpieza offline para candidatos de val (16-oct).
# Requiere que las dirs runs/eval_honest_val/<tag>/preds existan (cadena val).
cd "$HOME/projects/data-augh" || exit 1
PY="$HOME/miniconda3/envs/mioaug/bin/python"
for tag in "G-v2@tta" "G-v3@tta" "A-v3@tta" "A-v4@tta" \
           "A-v3+G-v2" "A-v4+G-v2" "A-v3+G-v3" "A-v4+G-v3"; do
  dir="runs/eval_honest_val/$tag"
  if [ ! -f "$dir/metrics.json" ]; then echo "SKIP $tag (sin metrics)"; continue; fi
  out="runs/eval_cleanup_val/${tag}.json"
  if [ -f "$out" ]; then echo "SKIP $tag (ya evaluado)"; continue; fi
  echo "== $tag"
  "$PY" -u src/cleanup_eval.py --run-dir "$dir" --grid "2,4000," --grid "2,8000," \
    --out "$out" || echo "FAIL $tag"
done
echo SWEEP_DONE
