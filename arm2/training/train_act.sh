#!/usr/bin/env bash
# Entrena una policy ACT con LeRobot sobre un LeRobotDataset convertido con convert_normacore.py
# y, opcionalmente, sube el checkpoint final a Hugging Face.
#
#   DATASET=<hf_user>/<nombre>_lerobot RUN=act_<skill> bash train_act.sh
#   DATASET=... RUN=... HF_REPO=<hf_user>/flexnsense_act_<skill>_<tarea> bash train_act.sh   # + subida
#   RUN=act_<skill> RESUME=1 bash train_act.sh                                               # reanudar
#
# Variables (todas por entorno):
#   DATASET        repo_id del LeRobotDataset (local en $HF_LEROBOT_HOME o en el Hub)   [obligatoria]
#   RUN            nombre del run = carpeta de checkpoints y del log                    [obligatoria]
#   HF_REPO        repo de modelo en HF donde subir el último checkpoint (vacío = no subir)
#   TASK           frase de la tarea, solo para la model card
#   EPISODES       lista de episodios a usar, p.ej. "[0,1,2,5]" (por defecto todos); el resto sirve de validación
#   HAND_MARKS=1   el dataset se convirtió con --hand-marks: deja hand.json en cada checkpoint
#   STEPS=30000  BATCH=32  CHUNK=30  N_ACTION_STEPS=15  SAVE_FREQ=5000  NUM_WORKERS=8
#   OUT_ROOT=./outputs/train   LOG_DIR=./outputs/logs
#   ENV_SH         script a "source" antes de empezar (activar venv, HF_HOME, ...)
#
# CHUNK: nº de acciones que predice ACT. A 10 Hz, 30 = 3 s (el ACT original usa ~2 s).
# N_ACTION_STEPS: cuántas ejecuta antes de re-planificar; solo afecta a inferencia.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "${ENV_SH:-}" ]] && source "$ENV_SH"

: "${RUN:?define RUN (nombre del run, p.ej. act_skill1_c30)}"
OUT_ROOT=$(realpath -m "${OUT_ROOT:-./outputs/train}"); LOG_DIR=$(realpath -m "${LOG_DIR:-./outputs/logs}")
OUT=$OUT_ROOT/$RUN; LOG=$LOG_DIR/$RUN.log
mkdir -p "$OUT_ROOT" "$LOG_DIR"

if [[ "${RESUME:-0}" == "1" ]]; then
  lerobot-train --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true \
    2>&1 | grep --line-buffered -v '^Svt' | tee -a "$LOG"
else
  : "${DATASET:?define DATASET (repo_id del LeRobotDataset)}"
  # push_to_hub de LeRobot va desactivado: la subida la hace upload_model.py al final.
  # En LeRobot 0.7.x el logging es Trackio (no wandb); aquí va desactivado, el log queda en $LOG.
  lerobot-train \
    --dataset.repo_id="$DATASET" \
    ${EPISODES:+--dataset.episodes="$EPISODES"} \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.chunk_size="${CHUNK:-30}" \
    --policy.n_action_steps="${N_ACTION_STEPS:-15}" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size="${BATCH:-32}" \
    --num_workers="${NUM_WORKERS:-8}" \
    --steps="${STEPS:-30000}" \
    --save_freq="${SAVE_FREQ:-5000}" \
    --log_freq=200 \
    --trackio.enable=false \
    --output_dir="$OUT" \
    --job_name="$RUN" \
    2>&1 | grep --line-buffered -v '^Svt' | tee "$LOG"
fi

if [[ "${HAND_MARKS:-0}" == "1" ]]; then
  for d in "$OUT"/checkpoints/*/pretrained_model; do [[ -L "$(dirname "$d")" ]] || echo '{"marks": true}' > "$d/hand.json"; done
fi
echo; echo "checkpoints: $OUT/checkpoints/<step>/pretrained_model   (último: $OUT/checkpoints/last/pretrained_model)"
# ── Subida a Hugging Face (solo si el entrenamiento terminó sin error: set -e + pipefail) ──
if [[ -n "${HF_REPO:-}" ]]; then
  python "$HERE/upload_model.py" --checkpoint "$OUT/checkpoints/last/pretrained_model" --repo-id "$HF_REPO" \
    ${TASK:+--task "$TASK"} 2>&1 | tee -a "$LOG"
else
  echo "HF_REPO vacío: no se sube. Para subirlo:"
  echo "  python $HERE/upload_model.py --checkpoint $OUT/checkpoints/last/pretrained_model --repo-id <hf_user>/<repo>"
fi
