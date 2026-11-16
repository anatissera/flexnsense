#!/usr/bin/env bash
# Fine-tune de SmolVLA (receta de NormaCore, smolvla_py) sobre un Parquet de normalize_cameras.py.
#
#   PARQUET=~/data/flexnsense_skill1_canon.parquet RUN=smolvla_flexnsense_skill1_c50 bash train_smolvla.sh
#
# Necesita NORMA_CORE (lo define env.sh). smolvla_py usa SU PROPIO venv (uv sync con su lock:
# transformers==5.3.0), distinto del de LeRobot; se crea aquí la primera vez.
# Variables: STEPS (12000), BATCH (64), SAVE_EVERY (2000), WORKERS (8), STD_FLOOR (0.02),
#            CAMERAS (side,gripper,table), OUT_ROOT (./outputs/train), LOG_DIR (./outputs/logs).
# No hay --resume en el train.py de NormaCore: si se corta, se relanza con otro RUN.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${NORMA_CORE:?define NORMA_CORE (source env.sh)}" "${PARQUET:?PARQUET=<parquet canónico>}" "${RUN:?RUN=<nombre del run>}"
OUT=$(realpath -m "${OUT_ROOT:-./outputs/train}/$RUN")
LOG_DIR=$(realpath -m "${LOG_DIR:-./outputs/logs}"); mkdir -p "$LOG_DIR"
PARQUET=$(realpath "$PARQUET")
[[ -e "$OUT" ]] && { echo "$OUT ya existe: usa otro RUN" >&2; exit 1; }

cd "$NORMA_CORE/software/ai/smolvla_py"
[[ -d .venv ]] || uv sync
.venv/bin/python -u "$HERE/train_smolvla.py" \
    --cameras "${CAMERAS:-side,gripper,table}" --std-floor "${STD_FLOOR:-0.02}" \
    --steps "${STEPS:-12000}" --batch-size "${BATCH:-64}" --save-every "${SAVE_EVERY:-2000}" \
    --num-workers "${WORKERS:-8}" --parquets "$PARQUET" --output "$OUT" \
    2>&1 | tee "$LOG_DIR/$RUN.log"
