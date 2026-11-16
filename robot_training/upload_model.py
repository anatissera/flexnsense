#!/usr/bin/env python
r"""Sube un checkpoint LeRobot (carpeta pretrained_model) a un repo de modelo de Hugging Face.

    python upload_model.py --checkpoint outputs/train/<RUN>/checkpoints/last/pretrained_model \
                           --repo-id <hf_user>/flexnsense_act_<skill>_<tarea> [--task "..."]
    python upload_model.py ... --dry-run      # enseña qué subiría, sin subir nada
    python upload_model.py ... --create       # crea el repo si no existe (privado; --public para público)

Los ficheros van a la RAÍZ del repo (config.json, model.safetensors, processors...), así
`ACTPolicy.from_pretrained("<repo_id>")` y `--policy.path=<repo_id>` funcionan tal cual.
También escribe una model card en README.md (se salta con --no-card).
Requiere `hf auth login` con permiso de escritura en el repo.

OJO: se sube también train_config.json (contiene rutas locales) y las estadísticas de
normalización del dataset. Si el repo es público, eso queda público.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

REQUIRED = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]


def model_card(repo_id: str, cfg: dict, train: dict, step: str, task: str | None, license_: str) -> str:
    feats = cfg.get("input_features", {})
    cams = [k for k in feats if k.startswith("observation.images.")]
    state_dim = (feats.get("observation.state", {}).get("shape") or ["?"])[0]
    act_dim = (cfg.get("output_features", {}).get("action", {}).get("shape") or ["?"])[0]
    ds = train.get("dataset", {})
    ptype = cfg.get("type", "policy")
    cam_rows = "\n".join(f"| `{k}` | {tuple(feats[k].get('shape', []))} | RGB camera |" for k in cams)
    return f"""---
license: {license_}
library_name: lerobot
pipeline_tag: robotics
tags:
- lerobot
- robotics
- {ptype}
- elrobot
- normacore
---

# {ptype.upper()} policy for the NormaCore ElRobot

[LeRobot](https://github.com/huggingface/lerobot) `{ptype}` policy for a NormaCore **ElRobot** follower arm
(7 joints + gripper, 8 Feetech ST3215 servos, IDs 1..8), trained from teleoperated demonstrations
recorded with NormaCore Station.

- **Task:** {f"`{task}`" if task else "see dataset"}
- **Checkpoint:** step {step} of {train.get("steps")} (batch size {train.get("batch_size")})
- **Training dataset (LeRobot format):** `{ds.get("repo_id")}`
- **chunk_size / n_action_steps:** {cfg.get("chunk_size")} / {cfg.get("n_action_steps")} (control at 10 Hz)

## Inputs / outputs

| Feature | Shape | Meaning |
|---|---|---|
| `observation.state` | ({state_dim},) | NormaCore `position_norm` in [0, 1], ordered by motor ID (last = gripper) |
{cam_rows}
| `action` | ({act_dim},) | NormaCore `goal_norm` in [0, 1], same order |

To drive the servos, map each action to raw ticks with the per-joint calibration range:
`ticks = range_min + goal_norm * (range_max - range_min)`.

NormaCore Station does not label the images of the `inference/normvla` queue and their order can
change between restarts: identify each camera by content before feeding the policy.

## Usage

```python
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

repo = "{repo_id}"
cfg = PreTrainedConfig.from_pretrained(repo)
policy = get_policy_class(cfg.type).from_pretrained(repo, config=cfg)
pre, post = make_pre_post_processors(cfg, pretrained_path=repo)
```

The normalization statistics are stored in the processor files of this repo.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="carpeta pretrained_model")
    p.add_argument("--repo-id", required=True, help="<hf_user>/<nombre_del_modelo>")
    p.add_argument("--task", default=None, help="frase de la tarea (para la model card)")
    p.add_argument("--license", default="mit")
    p.add_argument("--no-card", action="store_true", help="no tocar el README.md del repo")
    p.add_argument("--create", action="store_true", help="crear el repo si no existe")
    p.add_argument("--public", action="store_true", help="con --create: repo público (por defecto privado)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    src = Path(a.checkpoint).resolve()          # resuelve el symlink `last`
    missing = [f for f in REQUIRED if not (src / f).is_file()]
    if missing:
        sys.exit(f"{src} no parece un pretrained_model completo: faltan {missing}")
    step = src.parent.name
    cfg = json.loads((src / "config.json").read_text())
    train = json.loads((src / "train_config.json").read_text()) if (src / "train_config.json").exists() else {}
    files = sorted(f for f in src.iterdir() if f.is_file())
    total = sum(f.stat().st_size for f in files) / 1e6
    print(f"checkpoint: {src}  (step {step}, policy {cfg.get('type')})")
    print(f"destino:    https://huggingface.co/{a.repo_id}")
    for f in files:
        print(f"  {f.name:70s} {f.stat().st_size / 1e6:8.1f} MB")
    print(f"  total {total:.1f} MB" + ("" if a.no_card else "  + README.md (model card)"))
    if a.dry_run:
        print("dry-run: no se ha subido nada.")
        return

    api = HfApi()
    if a.create:
        api.create_repo(a.repo_id, repo_type="model", private=not a.public, exist_ok=True)
    info = api.model_info(a.repo_id)            # falla aquí si el repo no existe o no hay acceso
    print(f"repo {'PRIVADO' if info.private else 'PÚBLICO'}; subiendo ...")
    commit = api.upload_folder(repo_id=a.repo_id, repo_type="model", folder_path=str(src), path_in_repo=".",
                               commit_message=f"Upload {cfg.get('type')} checkpoint (step {step})")
    print("modelo:", getattr(commit, "commit_url", commit))
    if not a.no_card:
        card = model_card(a.repo_id, cfg, train, step, a.task, a.license)
        c = api.upload_file(repo_id=a.repo_id, repo_type="model", path_or_fileobj=card.encode(),
                            path_in_repo="README.md", commit_message=f"Model card (step {step})")
        print("card:  ", getattr(c, "commit_url", c))
    print(f"OK -> https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
