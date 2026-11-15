#!/usr/bin/env python
"""Policy y robot en la MISMA máquina (el portátil con Station).

    source <entorno con lerobot + norma-core>
    python run_local.py --checkpoint <.../pretrained_model> --dry-run      # primero en seco
    python run_local.py --checkpoint <.../pretrained_model>                # luego en serio

También sirve con el Station en otra máquina accesible por red: --station <host>.
"""

import argparse
import asyncio
import logging

from control import add_robot_args, run_control
from policy_runner import PolicyRunner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="carpeta pretrained_model del checkpoint LeRobot")
    p.add_argument("--device", default=None, help="cuda / cpu / mps (por defecto: cuda si hay)")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="acciones del chunk que se ejecutan antes de re-planificar (por defecto: el del checkpoint)")
    add_robot_args(p)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    runner = PolicyRunner(args.checkpoint, device=args.device, n_action_steps=args.n_action_steps)
    print(f"policy {runner.cfg.type} en {runner.device} | chunk {runner.chunk_size} | ejecuta {runner.n_action_steps}")

    async def get_chunk(state, jpeg_by_view, task):
        chunk, ms = await asyncio.get_running_loop().run_in_executor(
            None, runner.predict_from_jpegs, state, jpeg_by_view, task)
        return chunk, runner.n_action_steps, f"(inferencia {ms:.0f} ms)"

    asyncio.run(run_control(args, get_chunk, logging.getLogger("run_local")))


if __name__ == "__main__":
    main()
