#!/bin/bash
# protocol_val_chain.sh — seleccion de protocolo en el split VAL (16-oct-2026).
#
# Fase 1: finalistas individuales en val (sin TTA).
# Fase 2: TTA flips x4 de los finalistas en val.
# Los ensembles se deciden despues de ver el delta TTA y la rejilla de limpieza
# (src/cleanup_eval.py) sobre las preds guardadas.
#
# Guarda predicciones (--save-preds 1, def.) en runs/eval_honest_val/<tag>/preds/
# para la reevaluacion offline con limpieza por componentes.
#
# Idempotente: si runs/eval_honest_val/<tag>/metrics.json existe, lo salta.
# Lanzar (desprendido):
#   setsid nohup bash scripts/protocol_val_chain.sh > runs/val_chain_stdout.log 2>&1 < /dev/null &
cd "$HOME/projects/data-augh" || exit 1
PY="$HOME/miniconda3/envs/mioaug/bin/python"
LOG="runs/protocol_val_chain.log"
OUT="runs/eval_honest_val"
mkdir -p "$OUT" runs

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

run_one(){
  # tag cfg ckpt [args extra...]
  local tag="$1" cfg="$2" ckpt="$3"; shift 3
  if [ ! -f "$ckpt" ]; then log "SKIP $tag (sin checkpoint: $ckpt)"; return 0; fi
  if [ -f "$OUT/$tag/metrics.json" ]; then log "SKIP $tag (ya evaluado)"; return 0; fi
  log "START $tag"
  "$PY" -u src/eval_fullimage.py --cfg "$cfg" --ckpt "$ckpt" \
      --outdir "$OUT/$tag" --tag "$tag" --amp bf16 --batch 8 --split val \
      --save-preds 1 "$@" >> "runs/eval_honest_val_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "OK $tag"; else log "FAIL $tag rc=$rc"; fi
  return 0
}

run_ens(){
  # tag cfg_a ck_a cfg_b ck_b alpha [args extra...]
  local tag="$1" cfg_a="$2" ck_a="$3" cfg_b="$4" ck_b="$5" alpha="$6"; shift 6
  if [ ! -f "$ck_a" ] || [ ! -f "$ck_b" ]; then log "SKIP $tag (faltan checkpoints)"; return 0; fi
  if [ -f "$OUT/$tag/metrics.json" ]; then log "SKIP $tag (ya evaluado)"; return 0; fi
  log "START $tag (alpha=$alpha)"
  "$PY" -u src/eval_fullimage.py --cfg "$cfg_a" --ckpt "$ck_a" \
      --cfg-b "$cfg_b" --ckpt-b "$ck_b" --alpha "$alpha" \
      --outdir "$OUT/$tag" --tag "$tag" --amp bf16 --batch 8 --split val \
      --save-preds 1 "$@" >> "runs/eval_honest_val_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "OK $tag"; else log "FAIL $tag rc=$rc"; fi
  return 0
}

log "VAL CHAIN START (pid $$)"

# ---- Fase 1: finalistas en val (sin TTA) -----------------------------------
run_one G-v2 configs/seg_dpt_swin_v2.yaml checkpoints/exp_dpt_swin_v2/best.pt
run_one G-v3 configs/seg_dpt_swin_v3.yaml checkpoints/exp_dpt_swin_v3/best.pt
run_one A-v3 configs/seg_unet_a_v3.yaml   checkpoints/exp_unet_a_v3/best.pt
run_one A-v4 configs/seg_unet_a_v4.yaml   checkpoints/exp_unet_a_v4/best.pt

# ---- Fase 2: TTA flips x4 en val -------------------------------------------
run_one "G-v2@tta" configs/seg_dpt_swin_v2.yaml checkpoints/exp_dpt_swin_v2/best.pt --tta flips
run_one "G-v3@tta" configs/seg_dpt_swin_v3.yaml checkpoints/exp_dpt_swin_v3/best.pt --tta flips
run_one "A-v3@tta" configs/seg_unet_a_v3.yaml   checkpoints/exp_unet_a_v3/best.pt --tta flips
run_one "A-v4@tta" configs/seg_unet_a_v4.yaml   checkpoints/exp_unet_a_v4/best.pt --tta flips

# ---- Fase 3: ensembles (probs alpha 0.5) en val ----------------------------
run_ens "A-v3+G-v2" configs/seg_unet_a_v3.yaml checkpoints/exp_unet_a_v3/best.pt \
        configs/seg_dpt_swin_v2.yaml checkpoints/exp_dpt_swin_v2/best.pt 0.5
run_ens "A-v4+G-v2" configs/seg_unet_a_v4.yaml checkpoints/exp_unet_a_v4/best.pt \
        configs/seg_dpt_swin_v2.yaml checkpoints/exp_dpt_swin_v2/best.pt 0.5
run_ens "A-v3+G-v3" configs/seg_unet_a_v3.yaml checkpoints/exp_unet_a_v3/best.pt \
        configs/seg_dpt_swin_v3.yaml checkpoints/exp_dpt_swin_v3/best.pt 0.5
run_ens "A-v4+G-v3" configs/seg_unet_a_v4.yaml checkpoints/exp_unet_a_v4/best.pt \
        configs/seg_dpt_swin_v3.yaml checkpoints/exp_dpt_swin_v3/best.pt 0.5

log "VAL CHAIN DONE (fases 1-3)"
