#!/bin/bash
# orch_night.sh — cadena de la noche 15-16 oct 2026:
#   A-v3 -> eval -> G-v3 -> eval -> A-v4 -> eval -> 4 ensambles -> resumen
# Cada paso se registra en runs/night_status.log (START/OK/FAIL).
# Lanzar con: setsid nohup bash orch_night.sh > runs/orch_night_stdout.log 2>&1 &
cd "$HOME/projects/data-augh" || exit 1
export PYTHONUNBUFFERED=1
PY="$HOME/miniconda3/envs/mioaug/bin/python"
STATUS="runs/night_status.log"
mkdir -p runs

log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

run(){
  local tag=$1; shift
  log "START $tag"
  "$@" >> "runs/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "OK $tag"; else log "FAIL $tag rc=$rc"; fi
  return $rc
}

log "NIGHT START (orch_night.sh)"

run train_a_v3 "$PY" src/train_seg.py configs/seg_unet_a_v3.yaml
run eval_a_v3  "$PY" src/eval.py configs/seg_unet_a_v3.yaml --outdir=runs/eval_a_v3 --save-1panel

run train_g_v3 "$PY" src/train_seg.py configs/seg_dpt_swin_v3.yaml
run eval_g_v3  "$PY" src/eval.py configs/seg_dpt_swin_v3.yaml --outdir=runs/eval_g_v3 --save-1panel

run train_a_v4 "$PY" src/train_seg.py configs/seg_unet_a_v4.yaml
run eval_a_v4  "$PY" src/eval.py configs/seg_unet_a_v4.yaml --outdir=runs/eval_a_v4 --save-1panel

run ens_a3_g3 "$PY" src/ensemble_eval.py \
  --cfg-a configs/seg_unet_a_v3.yaml --ckpt-a checkpoints/exp_unet_a_v3/best.pt \
  --cfg-b configs/seg_dpt_swin_v3.yaml --ckpt-b checkpoints/exp_dpt_swin_v3/best.pt \
  --alpha 0.5 --outdir runs/ens_a3_g3 --save-1panel
run ens_a4_g3 "$PY" src/ensemble_eval.py \
  --cfg-a configs/seg_unet_a_v4.yaml --ckpt-a checkpoints/exp_unet_a_v4/best.pt \
  --cfg-b configs/seg_dpt_swin_v3.yaml --ckpt-b checkpoints/exp_dpt_swin_v3/best.pt \
  --alpha 0.5 --outdir runs/ens_a4_g3 --save-1panel
run ens_a3_g2 "$PY" src/ensemble_eval.py \
  --cfg-a configs/seg_unet_a_v3.yaml --ckpt-a checkpoints/exp_unet_a_v3/best.pt \
  --cfg-b configs/seg_dpt_swin_v2.yaml --ckpt-b checkpoints/exp_dpt_swin_v2/best.pt \
  --alpha 0.5 --outdir runs/ens_a3_g2 --save-1panel
run ens_a4_g2 "$PY" src/ensemble_eval.py \
  --cfg-a configs/seg_unet_a_v4.yaml --ckpt-a checkpoints/exp_unet_a_v4/best.pt \
  --cfg-b configs/seg_dpt_swin_v2.yaml --ckpt-b checkpoints/exp_dpt_swin_v2/best.pt \
  --alpha 0.5 --outdir runs/ens_a4_g2 --save-1panel

"$PY" src/night_summary.py >> "$STATUS" 2>&1
log "NIGHT DONE"
