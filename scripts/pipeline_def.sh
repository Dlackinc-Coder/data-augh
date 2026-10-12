#!/bin/bash
# Pipeline: espera B v4 -> eval B -> D (U-Net guided) -> E (SegFormer guided) -> F (LiteMedSAM guided)
# Cada fase: train (100 ep) + eval + copia de overlays al experimento
source ~/miniconda/etc/profile.d/conda.sh
conda activate mioaug
cd ~/data-augh || exit 1

STATUS=runs/pipeline_status.log
EVALLOG=runs/eval_results.log
echo "[$(date)] PIPELINE INICIADO" >> "$STATUS"

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

# --- Fase B: ya corriendo, esperar ---
echo "[$(date)] Esperando B v4..." >> "$STATUS"
if ! wait_for_train runs/train_segformer_v4.log configs/seg_segformer_mit_b2.yaml; then exit 1; fi
run_eval configs/seg_segformer_mit_b2.yaml experiments/exp_segformer_mit_b2 || exit 1

# --- Fase D: U-Net guided ---
run_train configs/seg_guided_unet.yaml runs/train_unet_guided.log || exit 1
run_eval configs/seg_guided_unet.yaml experiments/exp_unet_guided || exit 1

# --- Fase E: SegFormer guided ---
run_train configs/seg_guided_segformer.yaml runs/train_segformer_guided.log || exit 1
run_eval configs/seg_guided_segformer.yaml experiments/exp_segformer_guided || exit 1

# --- Fase F: LiteMedSAM guided ---
run_train configs/seg_guided_litemedsam.yaml runs/train_litemedsam_guided.log || exit 1
run_eval configs/seg_guided_litemedsam.yaml experiments/exp_litemedsam_guided || exit 1

echo "[$(date)] PIPELINE COMPLETO (B, D, E, F)" >> "$STATUS"
