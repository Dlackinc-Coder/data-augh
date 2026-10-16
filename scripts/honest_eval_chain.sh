#!/bin/bash
# honest_eval_chain.sh — Fase 2: re-evaluacion honesta full-image (16-oct-2026).
#
# Fase 1: checkpoints ya terminados (corre en paralelo a la cadena nocturna).
# Fase 2: espera a que termine orch_night.sh y evalua G-v3 y A-v4.
# Fase 3: TTA flips en los finalistas.
# Fase 4: ensembles honestos (media de probabilidades de A + G).
# Fase 5: resumen agregado (runs/eval_honest/SUMMARY.md + COMPARATIVA_HONESTA.md).
#
# Idempotente: si runs/eval_honest/<tag>/metrics.json ya existe, lo salta.
# Lanzar (desprendido):
#   setsid nohup bash scripts/honest_eval_chain.sh > runs/honest_chain_stdout.log 2>&1 < /dev/null &
cd "$HOME/projects/data-augh" || exit 1
PY="$HOME/miniconda3/envs/mioaug/bin/python"
LOG="runs/honest_eval_chain.log"
OUT="runs/eval_honest"
mkdir -p "$OUT" runs

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

run_one(){
  # tag cfg ckpt [args extra...]
  local tag="$1" cfg="$2" ckpt="$3"; shift 3
  if [ ! -f "$ckpt" ]; then log "SKIP $tag (sin checkpoint: $ckpt)"; return 0; fi
  if [ -f "$OUT/$tag/metrics.json" ]; then log "SKIP $tag (ya evaluado)"; return 0; fi
  log "START $tag"
  "$PY" -u src/eval_fullimage.py --cfg "$cfg" --ckpt "$ckpt" \
      --outdir "$OUT/$tag" --tag "$tag" --amp bf16 --batch 8 \
      --save-overlays 6 "$@" >> "runs/eval_honest_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "OK $tag"; else log "FAIL $tag rc=$rc"; fi
  return 0
}

run_ens(){
  # tag cfg_a ck_a cfg_b ck_b alpha
  local tag="$1" cfg_a="$2" ck_a="$3" cfg_b="$4" ck_b="$5" alpha="$6"
  if [ ! -f "$ck_a" ] || [ ! -f "$ck_b" ]; then log "SKIP $tag (faltan checkpoints)"; return 0; fi
  if [ -f "$OUT/$tag/metrics.json" ]; then log "SKIP $tag (ya evaluado)"; return 0; fi
  log "START $tag (alpha=$alpha)"
  "$PY" -u src/eval_fullimage.py --cfg "$cfg_a" --ckpt "$ck_a" \
      --cfg-b "$cfg_b" --ckpt-b "$ck_b" --alpha "$alpha" \
      --outdir "$OUT/$tag" --tag "$tag" --amp bf16 --batch 8 --save-overlays 6 \
      >> "runs/eval_honest_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "OK $tag"; else log "FAIL $tag rc=$rc"; fi
  return 0
}

log "HONEST CHAIN START (pid $$)"

# ---- Fase 1: checkpoints terminados (paralelo a la cadena nocturna) --------
run_one A-v2 configs/seg_unet_effb0_v2.yaml checkpoints/exp_unet_effb0_v2/best.pt
run_one A-v3 configs/seg_unet_a_v3.yaml     checkpoints/exp_unet_a_v3/best.pt
run_one G-v1 configs/seg_dpt_swin.yaml      checkpoints/exp_dpt_swin/best.pt
run_one G-v2 configs/seg_dpt_swin_v2.yaml   checkpoints/exp_dpt_swin_v2/best.pt
run_one H    configs/seg_dpt_pvtv2.yaml     checkpoints/exp_dpt_pvtv2/best.pt
run_one I    configs/seg_dpt_convnext.yaml  checkpoints/exp_dpt_convnext/best.pt
run_one J    configs/seg_dpt_vit.yaml       checkpoints/exp_dpt_vit/best.pt

# ---- Fase 2: esperar fin de la cadena nocturna ----------------------------
WAITED=0
while pgrep -f "orch_night" > /dev/null; do
  sleep 120
  WAITED=$((WAITED + 120))
  if [ $WAITED -ge 43200 ]; then log "AVISO: 12 h de espera, continuo igual"; break; fi
done
log "cadena nocturna terminada (espera ${WAITED}s)"
sleep 20

run_one G-v3 configs/seg_dpt_swin_v3.yaml checkpoints/exp_dpt_swin_v3/best.pt
run_one A-v4 configs/seg_unet_a_v4.yaml   checkpoints/exp_unet_a_v4/best.pt

# ---- Fase 3: TTA flips en finalistas --------------------------------------
run_one "A-v3@tta" configs/seg_unet_a_v3.yaml   checkpoints/exp_unet_a_v3/best.pt --tta flips
run_one "A-v4@tta" configs/seg_unet_a_v4.yaml   checkpoints/exp_unet_a_v4/best.pt --tta flips
run_one "G-v2@tta" configs/seg_dpt_swin_v2.yaml checkpoints/exp_dpt_swin_v2/best.pt --tta flips
run_one "G-v3@tta" configs/seg_dpt_swin_v3.yaml checkpoints/exp_dpt_swin_v3/best.pt --tta flips

# ---- Fase 4: ensembles honestos -------------------------------------------
run_ens "A-v3+G-v3" configs/seg_unet_a_v3.yaml checkpoints/exp_unet_a_v3/best.pt \
        configs/seg_dpt_swin_v3.yaml checkpoints/exp_dpt_swin_v3/best.pt 0.5
run_ens "A-v4+G-v3" configs/seg_unet_a_v4.yaml checkpoints/exp_unet_a_v4/best.pt \
        configs/seg_dpt_swin_v3.yaml checkpoints/exp_dpt_swin_v3/best.pt 0.5
run_ens "A-v3+G-v2" configs/seg_unet_a_v3.yaml checkpoints/exp_unet_a_v3/best.pt \
        configs/seg_dpt_swin_v2.yaml checkpoints/exp_dpt_swin_v2/best.pt 0.5
run_ens "A-v4+G-v2" configs/seg_unet_a_v4.yaml checkpoints/exp_unet_a_v4/best.pt \
        configs/seg_dpt_swin_v2.yaml checkpoints/exp_dpt_swin_v2/best.pt 0.5

# ---- Fase 5: resumen -------------------------------------------------------
"$PY" -u src/honest_summary.py >> "$LOG" 2>&1
log "HONEST CHAIN DONE"
