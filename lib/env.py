"""Read settings from the environment, falling back to a `.env` file (see .env.example).

    from lib.env import env_get
    bus = env_get("FOLLOWER_BUS_SERIAL")          # str or None
    cam = env_get("CAMERA_INDEX", default="0")
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """KEY=VALUE pairs from a .env file (quotes stripped, comments and blank lines skipped). {} if missing."""
    p = Path(path)
    out: dict[str, str] = {}
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def env_get(key: str, default: str | None = None, path: str | Path = ".env") -> str | None:
    """The environment variable `key`, else its value in `.env`, else `default`. Empty values count as unset."""
    val = os.environ.get(key)
    if not val:
        val = load_env(path).get(key)
    return val if val else default
