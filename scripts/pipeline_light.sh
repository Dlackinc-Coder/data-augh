#!/bin/bash
# Pipeline: DeepLabV3+ -> FPN -> MAnet (512², random sampling)
# Cada fase: train (100 ep) + eval (random) + copia de overlays
source ~/miniconda/etc/profile.d/conda.sh
conda activate mioaug
cd ~/data-augh || exit 1

STATUS=runs/pipeline_light_status.log
EVALLOG=runs/eval_light_results.log
echo "[$(date)] PIPELINE LIGHT INICIADO" >> "$STATUS"

log_has() { grep -q "$2" "$1" 2>/dev/null; }
proc_alive() { pgrep -f "train_seg.py $1" > /dev/null; }

wait_for_train() {  # $1=log $2=cfg
  while true; do
    if log_has "$1" "^done"; then return 0; fi
    if ! proc_alive "$2"; then
      echo "[$(date)] FALLO: $2 termino sin 'done' (ver $1)" >> "$STATUS"
      tail -5 "$1" >> "$STATUS"
      return 1
    fi
    sleep 60
  done
}

run_eval() {  # $1=cfg $2=exp_dir
  mkdir -p "$2"
  echo "[$(date)] EVAL $1" >> "$STATUS"
  if ! python src/eval.py "$1" --save-1panel >> "$EVALLOG" 2>&1; then
    echo "[$(date)] FALLO eval $1" >> "$STATUS"
    return 1
  fi
  rm -rf "$2/test_overlays" "$2/test_overlays_1panel"
  cp -r runs/test_overlays "$2/test_overlays" 2>/dev/null
  cp -r runs/test_overlays_1panel "$2/test_overlays_1panel" 2>/dev/null
  return 0
}

run_train() {  # $1=cfg $2=log
  echo "[$(date)] TRAIN $1 -> $2" >> "$STATUS"
  python src/train_seg.py "$1" > "$2" 2>&1
  wait_for_train "$2" "$1"
}

# --- Fase 1: DeepLabV3+ ---
run_train configs/seg_deeplabv3plus.yaml runs/train_deeplabv3plus.log || exit 1
run_eval configs/seg_deeplabv3plus.yaml experiments/exp_deeplabv3plus || exit 1

# --- Fase 2: FPN ---
run_train configs/seg_fpn.yaml runs/train_fpn.log || exit 1
run_eval configs/seg_fpn.yaml experiments/exp_fpn || exit 1

# --- Fase 3: MAnet ---
run_train configs/seg_manet.yaml runs/train_manet.log || exit 1
run_eval configs/seg_manet.yaml experiments/exp_manet || exit 1

echo "[$(date)] PIPELINE LIGHT COMPLETO (DeepLabV3+, FPN, MAnet)" >> "$STATUS"
