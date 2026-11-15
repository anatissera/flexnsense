#!/usr/bin/env bash
# Prepara el entorno de entrenamiento/inferencia: venv Python 3.12 + LeRobot (extras de ACT y
# SmolVLA) + norma-core (cliente station_py y protobufs de Python).
#
#   bash setup_env.sh [DIR]        # DIR por defecto: ./robot_env   (necesita uv y git)
#   source DIR/env.sh              # en cada terminal nueva
#
# Probado con: LeRobot 0.7.1 (commit d490e472), torch 2.15 + CUDA, norma-core main == v0.1.0-beta.10
# en protobufs/station_py/smolvla_py, Ubuntu 24.04, RTX PRO 4000 Blackwell.
set -euo pipefail
DIR=$(realpath -m "${1:-./robot_env}"); mkdir -p "$DIR"; cd "$DIR"
LEROBOT_REF=${LEROBOT_REF:-d490e472}

[[ -d venv ]] || uv venv --python 3.12 venv
source venv/bin/activate

[[ -d lerobot ]] || git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout -q "$LEROBOT_REF"
# training = accelerate + trackio; dataset = datasets/pyarrow/av/torchcodec; smolvla = transformers
uv pip install -e "lerobot[training,dataset,smolvla]"

# norma-core: sparse clone. target/gen_python DEBE estar en el cone: si no, cualquier
# `git sparse-checkout add` posterior borra los protobufs generados (están en .gitignore).
if [[ ! -d norma-core ]]; then
  git clone --filter=blob:none --sparse --depth 1 https://github.com/norma-core/norma-core.git
  git -C norma-core sparse-checkout set software/station/shared software/station/examples \
      software/station/bin/station software/ai/smolvla_py shared/gremlin_py protobufs \
      hardware/elrobot/simulation target/gen_python
fi
( cd norma-core
  for d in protobufs/*/; do n=$(basename "$d")
    python shared/gremlin_py/gremlin.py --proto-root "$d" --target-root "target/gen_python/protobuf/$n" \
      --project-root . --gremlin-import-path shared.gremlin_py.gremlin >/dev/null
  done )

# .pth: raíz del repo (software.*, target.*), station_py a secas y el port smolvla de NormaCore.
# (smolvla_py NO se instala con pip: fija transformers==5.3.0 y chocaría con LeRobot.)
SP=$(python -c "import site; print(site.getsitepackages()[0])")
printf '%s\n%s\n%s\n' "$DIR/norma-core" "$DIR/norma-core/software/station/shared" \
  "$DIR/norma-core/software/ai/smolvla_py" > "$SP/normacore.pth"

cat > env.sh <<EOS
# source $DIR/env.sh
source $DIR/venv/bin/activate
export NORMA_CORE=$DIR/norma-core
export HF_HOME=\${HF_HOME:-$DIR/cache/huggingface}   # token, hub cache y LeRobotDatasets (\$HF_HOME/lerobot)
export TORCH_HOME=\${TORCH_HOME:-$DIR/cache/torch}
EOS

python - <<'PY'
import torch
from station_py import new_station_client
from target.gen_python.protobuf.drivers.inferences import normvla
from lerobot.policies.act.configuration_act import ACTConfig
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU")
print("lerobot + station_py + protobufs OK")
PY
echo; echo "Listo. En cada terminal:  source $DIR/env.sh     (y una vez:  hf auth login)"
