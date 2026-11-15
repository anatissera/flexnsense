"""Carga una policy LeRobot (ACT u otra) desde un checkpoint y predice chunks de acciones.

Misma carga que lerobot/rollout/context.py (LeRobot 0.7.1):
  PreTrainedConfig.from_pretrained -> get_policy_class(...).from_pretrained
  make_pre_post_processors(pretrained_path=..., device_processor override)
La normalización (mean/std del dataset) va dentro de los processors guardados en el checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from station_io import VIEWS, decode_rgb


class PolicyRunner:
    def __init__(self, checkpoint: str | Path, device: str | None = None, n_action_steps: int | None = None):
        path = str(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            # Sin esto torch usa todos los cores y, con otra carga, la latencia se dispara
            # (medido aquí: 3.3 s con hilos por defecto vs 30-45 ms con 4-8 hilos).
            import os
            torch.set_num_threads(int(os.environ.get("POLICY_THREADS", min(8, os.cpu_count() or 4))))
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = self.device
        if n_action_steps is not None and hasattr(cfg, "n_action_steps"):
            cfg.n_action_steps = n_action_steps
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.image_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
        expected = [f"observation.images.{v}" for v in VIEWS]
        if sorted(self.image_keys) != sorted(expected):
            raise RuntimeError(f"la policy espera cámaras {self.image_keys}, este código da {expected}")
        self.chunk_size = int(getattr(cfg, "chunk_size", 1))
        self.n_action_steps = int(getattr(cfg, "n_action_steps", self.chunk_size))

    @torch.no_grad()
    def predict_chunk(self, state: np.ndarray, rgb_by_view: dict[str, np.ndarray], task: str = "") -> np.ndarray:
        """state (8,) position_norm; rgb_by_view: {vista: HxWx3 uint8}. Devuelve (chunk_size, 8) goal_norm."""
        obs = {"observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32))}
        for v, img in rgb_by_view.items():
            obs[f"observation.images.{v}"] = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True)).permute(2, 0, 1).float() / 255.0
        obs["task"] = task
        batch = self.pre(obs)
        actions = self.policy.predict_action_chunk(batch)          # (1, chunk, 8) normalizado
        actions = self.post(actions)                                # desnormalizado -> goal_norm
        return actions.squeeze(0).float().cpu().numpy()

    def predict_from_jpegs(self, state: np.ndarray, jpeg_by_view: dict[str, bytes], task: str = "") -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        chunk = self.predict_chunk(state, {v: decode_rgb(j) for v, j in jpeg_by_view.items()}, task)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return chunk, (time.perf_counter() - t0) * 1000
