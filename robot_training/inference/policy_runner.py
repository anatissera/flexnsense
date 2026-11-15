"""Carga una policy desde un checkpoint y predice chunks de acciones. Dos backends:

  * PolicyRunner    - checkpoint LeRobot (ACT u otra): carpeta `pretrained_model`.
  * SmolVLARunner   - checkpoint del fine-tune de NormaCore (train_smolvla.py): carpeta con
                      config.json, model.safetensors, stats.safetensors y elrobot.json.

`load_runner(checkpoint)` elige el backend mirando si existe `elrobot.json`.

PolicyRunner usa la misma carga que lerobot/rollout/context.py (LeRobot 0.7.1):
  PreTrainedConfig.from_pretrained -> get_policy_class(...).from_pretrained
  make_pre_post_processors(pretrained_path=..., device_processor override)
La normalización (mean/std del dataset) va dentro de los processors guardados en el checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

import json
import os
from types import SimpleNamespace

from station_io import VIEWS, decode_rgb


def _pick_device(device: str | None) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        # Sin esto torch usa todos los cores y, con otra carga, la latencia se dispara
        # (medido aquí: 3.3 s con hilos por defecto vs 30-45 ms con 4-8 hilos).
        torch.set_num_threads(int(os.environ.get("POLICY_THREADS", min(8, os.cpu_count() or 4))))
    return device


class _JpegMixin:
    n_obs_steps = 1   # >1: la policy necesita también la observación del tick anterior (`prev`)
    state_dim = 8     # 13 = 8 joints + 5 de las yemas de los dedos (hand_features.py); las añade control.py

    def predict_from_jpegs(self, state: np.ndarray, jpeg_by_view: dict[str, bytes], task: str = "",
                           prev: tuple[np.ndarray, dict[str, bytes]] | None = None) -> tuple[np.ndarray, float]:
        """`prev` = (state, jpeg_by_view) del frame anterior de Station; solo se usa si n_obs_steps > 1."""
        t0 = time.perf_counter()
        rgb = {v: decode_rgb(j) for v, j in jpeg_by_view.items()}
        if self.n_obs_steps > 1:
            prev_rgb = (prev[0], {v: decode_rgb(j) for v, j in prev[1].items()}) if prev is not None else None
            chunk = self.predict_chunk(state, rgb, task, prev=prev_rgb)
        else:
            chunk = self.predict_chunk(state, rgb, task)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return chunk, (time.perf_counter() - t0) * 1000


class PolicyRunner(_JpegMixin):
    def __init__(self, checkpoint: str | Path, device: str | None = None, n_action_steps: int | None = None):
        # import perezoso: el backend SmolVLA no necesita LeRobot instalado
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        path = str(checkpoint)
        self.device = _pick_device(device)
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
        self.n_action_steps = int(getattr(cfg, "n_action_steps", getattr(cfg, "chunk_size", 1)))
        # Diffusion Policy no tiene chunk_size: predice `horizon` y devuelve n_action_steps acciones.
        self.chunk_size = int(getattr(cfg, "chunk_size", self.n_action_steps))
        self.n_obs_steps = int(getattr(cfg, "n_obs_steps", 1))
        self.state_dim = int(cfg.input_features["observation.state"].shape[0])

    def _batch(self, state: np.ndarray, rgb_by_view: dict[str, np.ndarray], task: str) -> dict:
        obs = {"observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32))}
        for v, img in rgb_by_view.items():
            obs[f"observation.images.{v}"] = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True)).permute(2, 0, 1).float() / 255.0
        obs["task"] = task
        return self.pre(obs)

    @torch.no_grad()
    def predict_chunk(self, state: np.ndarray, rgb_by_view: dict[str, np.ndarray], task: str = "",
                      prev: tuple[np.ndarray, dict[str, np.ndarray]] | None = None) -> np.ndarray:
        """state (8,) position_norm; rgb_by_view: {vista: HxWx3 uint8}. Devuelve (T, 8) goal_norm.

        Con n_obs_steps > 1 (Diffusion Policy) la policy recibe [anterior, actual], igual que en el
        entrenamiento (frames consecutivos a 10 Hz). Sin `prev` (primer tick) se repite el actual,
        como hace LeRobot al empezar un episodio.
        """
        batch = self._batch(state, rgb_by_view, task)
        if self.n_obs_steps > 1:
            old = self._batch(prev[0], prev[1], task) if prev is not None else batch
            keys = ["observation.state", *self.image_keys]
            hist = [old] * (self.n_obs_steps - 1) + [batch]
            batch = {**batch, **{k: torch.stack([h[k] for h in hist], dim=1) for k in keys}}
        actions = self.policy.predict_action_chunk(batch)          # (1, chunk, 8) normalizado
        actions = self.post(actions)                                # desnormalizado -> goal_norm
        return actions.squeeze(0).float().cpu().numpy()


class SmolVLARunner(_JpegMixin):
    """SmolVLA de NormaCore (paquete `smolvla` de norma-core/software/ai/smolvla_py).

    Hace lo mismo que su scripts/run_policy.py (normaliza el estado con stats.safetensors,
    tokeniza la tarea, predict_action_chunk, desnormaliza), pero devuelve el chunk entero y
    coloca cada cámara por NOMBRE de vista según elrobot.json, no por índice de images[].
    """

    DEFAULT_N_ACTION_STEPS = 15   # 1.5 s a 10 Hz de un chunk de 50

    def __init__(self, checkpoint: str | Path, device: str | None = None, n_action_steps: int | None = None):
        from smolvla import SmolVLAPolicy
        from smolvla.normalize import normalize_state, unnormalize_action
        from smolvla.stats import load_stats

        path = Path(checkpoint)
        meta = json.loads((path / "elrobot.json").read_text())
        if sorted(meta["views"]) != sorted(VIEWS):
            raise RuntimeError(f"la policy espera cámaras {meta['views']}, este código da {list(VIEWS)}")
        self.key_of_view = dict(zip(meta["views"], meta["image_keys"]))
        self.default_task = meta.get("task", "")
        self.device = _pick_device(device)
        self.stats = {k: v.to(self.device) for k, v in load_stats(path / "stats.safetensors").items()}
        self.policy = SmolVLAPolicy.from_pretrained(
            path, config_overrides={"load_vlm_weights": False}, strict=False).to(self.device).eval()
        self._normalize_state, self._unnormalize_action = normalize_state, unnormalize_action
        self.cfg = SimpleNamespace(type="smolvla")
        self.chunk_size = int(self.policy.config.chunk_size)
        self.n_action_steps = min(int(n_action_steps or self.DEFAULT_N_ACTION_STEPS), self.chunk_size)
        self._tokens: dict[str, tuple] = {}

    @torch.no_grad()
    def predict_chunk(self, state: np.ndarray, rgb_by_view: dict[str, np.ndarray], task: str = "") -> np.ndarray:
        """state (8,) position_norm; rgb_by_view: {vista: HxWx3 uint8}. Devuelve (chunk_size, 8) goal_norm."""
        st = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        batch = {"observation.state": self._normalize_state(st, self.stats)}
        for v, img in rgb_by_view.items():
            batch[self.key_of_view[v]] = (torch.from_numpy(np.array(img, dtype=np.uint8, copy=True))
                                          .permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0)
        task = task or self.default_task     # sin --task: la frase con la que se entrenó
        if not task:
            raise RuntimeError("SmolVLA necesita la instrucción: pasa --task \"<frase del dataset>\"")
        if task not in self._tokens:
            self._tokens[task] = self.policy.tokenize_task(task, device=self.device)
        batch["observation.language.tokens"], batch["observation.language.attention_mask"] = self._tokens[task]
        # autocast bf16 como en el entrenamiento: el VLM se carga en bf16 y sin autocast la
        # atención falla por mezclar bf16 con float32 (le pasa al run_policy.py original).
        with torch.autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
            pred = self.policy.predict_action_chunk(batch)[0]             # (chunk, 8) normalizado
        return self._unnormalize_action(pred.float(), self.stats).cpu().numpy()


def load_runner(checkpoint: str | Path, device: str | None = None, n_action_steps: int | None = None):
    cls = SmolVLARunner if (Path(checkpoint) / "elrobot.json").exists() else PolicyRunner
    return cls(checkpoint, device=device, n_action_steps=n_action_steps)
