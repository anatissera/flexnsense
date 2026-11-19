#!/usr/bin/env bash
# Entrena Diffusion Policy con LeRobot sobre un LeRobotDataset convertido con convert_normacore.py
# y, opcionalmente, sube el checkpoint final a Hugging Face. Mismo uso que train_act.sh:
#
#   DATASET=<hf_user>/<nombre>_lerobot RUN=diffusion_<skill>_h16 bash train_diffusion.sh
#   DATASET=... RUN=... HF_REPO=<hf_user>/flexnsense_diffusion_<skill>_<tarea> bash train_diffusion.sh
#   RUN=diffusion_<skill>_h16 RESUME=1 bash train_diffusion.sh                               # reanudar
#
# Variables (todas por entorno):
#   DATASET, RUN [obligatorias]   HF_REPO, TASK (subida y model card)   ENV_SH
#   STEPS=40000  BATCH=64  SAVE_FREQ=10000  NUM_WORKERS=8
#   HORIZON=16  N_ACTION_STEPS=8  N_OBS_STEPS=2  CROP=200  INFER_STEPS=10
#   OUT_ROOT=./outputs/train   LOG_DIR=./outputs/logs
#
# Los valores por defecto de LeRobot (horizon 64, n_action_steps 32) están pensados para 30 Hz.
# A 10 Hz: HORIZON=16 predice 1.6 s y N_ACTION_STEPS=8 ejecuta 0.8 s antes de re-planificar
# (los del Diffusion Policy original). HORIZON debe ser múltiplo de 8 (U-Net).
# N_OBS_STEPS=2: la policy ve el frame actual y el anterior; inference/ ya se lo da.
# CROP: recorte aleatorio CROPxCROP de las imágenes 224x224 al entrenar (centrado en inferencia).
# INFER_STEPS: pasos de denoising con DDIM en inferencia (se entrena con 100).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "${ENV_SH:-}" ]] && source "$ENV_SH"

: "${RUN:?define RUN (nombre del run, p.ej. diffusion_skill1_h16)}"
OUT_ROOT=$(realpath -m "${OUT_ROOT:-./outputs/train}"); LOG_DIR=$(realpath -m "${LOG_DIR:-./outputs/logs}")
OUT=$OUT_ROOT/$RUN; LOG=$LOG_DIR/$RUN.log
mkdir -p "$OUT_ROOT" "$LOG_DIR"

if [[ "${RESUME:-0}" == "1" ]]; then
  lerobot-train --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true \
    2>&1 | grep --line-buffered -v '^Svt' | tee -a "$LOG"
else
  : "${DATASET:?define DATASET (repo_id del LeRobotDataset)}"
  HORIZON=${HORIZON:-16}; N_ACTION_STEPS=${N_ACTION_STEPS:-8}; N_OBS_STEPS=${N_OBS_STEPS:-2}; CROP=${CROP:-200}
  lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.image_transforms.enable=true \
    --policy.type=diffusion \
    --policy.n_obs_steps="$N_OBS_STEPS" \
    --policy.horizon="$HORIZON" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --policy.drop_n_last_frames=$((HORIZON - N_ACTION_STEPS - N_OBS_STEPS + 1)) \
    --policy.crop_shape="[$CROP,$CROP]" \
    --policy.noise_scheduler_type=DDIM \
    --policy.num_inference_steps="${INFER_STEPS:-10}" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size="${BATCH:-64}" \
    --num_workers="${NUM_WORKERS:-8}" \
    --steps="${STEPS:-40000}" \
    --save_freq="${SAVE_FREQ:-10000}" \
    --log_freq=200 \
    --trackio.enable=false \
    --output_dir="$OUT" \
    --job_name="$RUN" \
    2>&1 | grep --line-buffered -v '^Svt' | tee "$LOG"
fi

echo; echo "checkpoints: $OUT/checkpoints/<step>/pretrained_model   (último: $OUT/checkpoints/last/pretrained_model)"
if [[ -n "${HF_REPO:-}" ]]; then
  python "$HERE/upload_model.py" --checkpoint "$OUT/checkpoints/last/pretrained_model" --repo-id "$HF_REPO" \
    ${TASK:+--task "$TASK"} 2>&1 | tee -a "$LOG"
else
  echo "HF_REPO vacío: no se sube. Para subirlo:"
  echo "  python $HERE/upload_model.py --checkpoint $OUT/checkpoints/last/pretrained_model --repo-id <hf_user>/<repo>"
fi
