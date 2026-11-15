"""E/S con NormaCore Station para el ElRobot (sin torch: solo numpy + PIL + station_py).

Lo usan tanto `run_local.py` (policy en el portátil) como `robot_client.py` (policy en
un servidor remoto). Basado en software/ai/smolvla_py/scripts/run_policy.py y en
software/station/examples/st3215-remote-teleop-py de norma-core.

Hechos verificados en el código de norma-core (v0.1.0-beta.10 == main en estas rutas):
  * Cola `inference/normvla` (formato normvla, update-interval 100 ms = 10 Hz).
    Solo emite frames si UN único bus tiene torque (modo st3215-bus: "auto") y hay imágenes.
    Joints ordenados por motor ID. `normvla.Image` NO lleva ID de cámara: el orden de
    `images[]` es el orden de registro de las colas usbvideo (puede cambiar al reiniciar).
  * Cola `st3215/inference`: estado de todos los buses aunque no haya torque.
  * Comandos: StationCommandsPack -> cola "commands"; ST3215 sync_write a un registro.
    Registros (RAM): TORQUE_ENABLE 0x28, GOAL_POSITION 0x2A, PRESENT_POSITION 0x38.
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
import sys
import time

# En una máquina sin el .pth (p.ej. el portátil): export NORMA_CORE=/ruta/a/norma-core
if os.environ.get("NORMA_CORE"):
    _nc = os.environ["NORMA_CORE"]
    sys.path[:0] = [_nc, os.path.join(_nc, "software", "station", "shared")]
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from station_py import new_station_client, send_commands  # noqa: F401  (re-export)
from target.gen_python.protobuf.drivers.inferences import normvla
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.station import commands, drivers

NORMVLA_QUEUE = "inference/normvla"
ST3215_QUEUE = "st3215/inference"
RAM_TORQUE_ENABLE = 0x28
RAM_GOAL_POSITION = 0x2A
RAM_PRESENT_POSITION = 0x38
MAX_ANGLE_STEP = 4095
SIGN_BIT_MASK = 0x8000

from safety import GoalSafety, goal_norm_to_ticks  # noqa: E402,F401  (re-export)

VIEWS = ("side", "gripper", "table")  # mismo orden y nombres que el LeRobotDataset convertido
DEFAULT_REFS = Path(__file__).with_name("camera_refs.npz")


# ─────────────────────────────── comandos ST3215 ───────────────────────────────

def sync_write(bus_serial: str, address: int, motors: list[tuple[int, bytes]]) -> commands.DriverCommand:
    cmd = st3215.Command(
        target_bus_serial=bus_serial,
        sync_write=st3215.ST3215SyncWriteCommand(
            address=address,
            motors=[st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=m, value=v) for m, v in motors],
        ),
    )
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())


def goal_command(bus_serial: str, motor_ids: list[int], raw_goals: list[int]) -> commands.DriverCommand:
    return sync_write(bus_serial, RAM_GOAL_POSITION,
                      [(m, int(r).to_bytes(2, "little")) for m, r in zip(motor_ids, raw_goals)])


async def set_torque(client, bus_serial: str, motor_ids: list[int], enable: bool) -> None:
    v = b"\x01" if enable else b"\x00"
    await send_commands(client, [sync_write(bus_serial, RAM_TORQUE_ENABLE, [(m, v) for m in motor_ids])])


# ─────────────────────────────── lectura de estado ─────────────────────────────

def _normal_position(raw: int) -> int:
    if raw & SIGN_BIT_MASK:
        return (MAX_ANGLE_STEP + 1 - (raw & MAX_ANGLE_STEP)) & MAX_ANGLE_STEP
    return raw & MAX_ANGLE_STEP


async def read_bus_positions(client, bus_serial: str, motor_ids: list[int], timeout: float = 5.0) -> dict[int, int]:
    """Posición actual (ticks) de cada motor del bus, vía st3215/inference (funciona sin torque)."""
    deadline = time.monotonic() + timeout
    seen = []
    while time.monotonic() < deadline:
        qr = client.read_from_tail(ST3215_QUEUE, offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=max(0.1, deadline - time.monotonic()))
        if entry is None:
            await asyncio.sleep(0.05)
            continue
        state = st3215.InferenceStateReader(entry.Data)
        for bus in state.get_buses() or []:
            info = bus.get_bus()
            serial = info.get_serial_number() if info else ""
            if serial not in seen:
                seen.append(serial)
            if serial != bus_serial:
                continue
            pos = {}
            for m in bus.get_motors() or []:
                sb = bytes(m.get_state())
                if len(sb) >= RAM_PRESENT_POSITION + 2:
                    pos[m.get_id()] = _normal_position(struct.unpack_from("<H", sb, RAM_PRESENT_POSITION)[0])
            missing = [i for i in motor_ids if i not in pos]
            if missing:
                raise RuntimeError(f"bus {bus_serial}: faltan motores {missing} (hay {sorted(pos)})")
            return pos
    raise RuntimeError(f"bus {bus_serial} no encontrado en {ST3215_QUEUE} (buses vistos: {seen})")


async def safe_enable_torque(client, bus_serial: str, motor_ids: list[int]) -> None:
    """Escribe goal = posición actual y DESPUÉS activa el torque (evita un salto al arrancar)."""
    pos = await read_bus_positions(client, bus_serial, motor_ids)
    await send_commands(client, [goal_command(bus_serial, motor_ids, [pos[m] for m in motor_ids])])
    await set_torque(client, bus_serial, motor_ids, True)


# ─────────────────────────────── frames normvla ────────────────────────────────

class FrameShapeError(RuntimeError):
    """Station entrega un nº de joints o de imágenes distinto del esperado.

    Station pone una imagen por cada cámara que tiene activa (y descarta el frame entero si
    alguna va con retraso), así que menos imágenes = una cámara que Station no tiene abierta.
    """

    def __init__(self, n_joints: int, exp_joints: int, jpegs: list[bytes], exp_images: int):
        self.jpegs, self.exp_images = jpegs, exp_images
        super().__init__(f"frame con {n_joints} joints / {len(jpegs)} imágenes; se esperaban {exp_joints}/{exp_images}")


@dataclass
class Obs:
    frame_id: bytes
    state: np.ndarray          # (8,) position_norm
    positions: np.ndarray      # (8,) ticks actuales
    ranges: list[tuple[int, int]]
    jpegs: list[bytes]         # en el orden de slots de Station (NO de vistas)


async def fetch_frame(client, timeout: float) -> normvla.FrameReader:
    qr = client.read_from_tail(NORMVLA_QUEUE, offset=b"\x00", limit=1, step=1, buf_size=1)
    entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
    if entry is None:
        raise RuntimeError(f"{NORMVLA_QUEUE} cerrada sin frame ({qr.err})")
    return normvla.FrameReader(memoryview(bytes(entry.Data)))


async def fetch_fresh_obs(client, last_id: bytes, timeout: float, n_joints: int = 8, n_images: int = 3) -> Obs:
    """Espera un frame nuevo (global_frame_id distinto). Station lo publica a ~10 Hz."""
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"sin frames nuevos en {NORMVLA_QUEUE} en {timeout}s. ¿El follower tiene torque? "
                "¿El leader está SIN torque? (con st3215-bus: auto, Station salta frames si 0 o 2 buses tienen torque)")
        try:
            fr = await fetch_frame(client, timeout)
        except asyncio.TimeoutError:
            continue
        fid = bytes(fr.get_global_frame_id())
        if fid == last_id:
            await asyncio.sleep(0.01)
            continue
        joints = fr.get_joints() or []
        images = fr.get_images() or []
        if len(joints) != n_joints or len(images) != n_images:
            raise FrameShapeError(len(joints), n_joints, [bytes(im.get_jpeg()) for im in images], n_images)
        return Obs(
            frame_id=fid,
            state=np.array([j.get_position_norm() for j in joints], dtype=np.float32),
            positions=np.array([j.get_position() for j in joints], dtype=np.int64),
            ranges=[(int(j.get_range_min()), int(j.get_range_max())) for j in joints],
            jpegs=[bytes(im.get_jpeg()) for im in images],
        )


# ─────────────────────────────── cámaras ───────────────────────────────────────

def _feat(jpeg: bytes) -> np.ndarray:
    a = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L").resize((32, 32)), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


class CameraMapper:
    """Asigna cada slot de imagen de Station a una vista (side/gripper/table).

    Banco de ejemplos por vista (camera_refs.npz, de los 43 episodios). Para cada slot y
    vista: media de las top-k similitudes con el banco; se elige la PERMUTACIÓN completa de
    mayor puntuación y se exige margen frente a la segunda. Leave-one-episode-out en el
    dataset: 43/43 correctos, margen mínimo 0.276. Se decide al arrancar y se fija; también
    se puede forzar con `fixed_order`.
    """

    PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    def __init__(self, refs_path: Path = DEFAULT_REFS, fixed_order: list[str] | None = None,
                 min_margin: float = 0.10, k: int = 5):
        self.min_margin, self.k, self.margin = min_margin, k, float("nan")
        self.slot_of_view: dict[str, int] | None = None
        if fixed_order:
            assert sorted(fixed_order) == sorted(VIEWS), f"--camera-order debe ser una permutación de {VIEWS}"
            self.slot_of_view = {v: i for i, v in enumerate(fixed_order)}
        have_refs = Path(refs_path).exists()
        if not have_refs and not fixed_order:
            raise SystemExit(f"falta {refs_path}: genéralo con  python make_camera_refs.py --src <dataset_normacore>  "
                             "(o fuerza el orden con --camera-order)")
        z = np.load(refs_path) if have_refs else {}
        self.bank = [z[v].astype(np.float32) for v in VIEWS] if len(z) else []   # lista de (N_v, 1024)
        self._S = np.zeros((3, 3))
        self._n = 0

    @property
    def locked(self) -> bool:
        return self.slot_of_view is not None

    def scores(self, jpegs: list[bytes]) -> np.ndarray:
        """(slot, vista): media de las top-k similitudes con el banco de esa vista."""
        S = np.zeros((3, 3))
        for s, j in enumerate(jpegs):
            f = _feat(j)
            for v, b in enumerate(self.bank):
                sims = b @ f / f.size
                S[s, v] = np.sort(sims)[-self.k:].mean()
        return S

    def observe(self, jpegs: list[bytes], need: int = 10) -> bool:
        """Acumula frames; cuando hay `need`, decide y fija la asignación. Devuelve True al fijarla."""
        if self.locked:
            return True
        self._S += self.scores(jpegs)
        self._n += 1
        if self._n < need:
            return False
        S = self._S / self._n
        ranked = sorted(((sum(S[s, p[s]] for s in range(3)), p) for p in self.PERMS), reverse=True)
        (best, perm), (second, _) = ranked[0], ranked[1]
        margin = (best - second) / 3
        if margin < self.min_margin:
            raise RuntimeError(
                f"No puedo asignar las cámaras con seguridad (perm={perm}, margen={margin:.3f} < {self.min_margin}).\n"
                f"puntuación slot x {VIEWS}:\n{np.round(S, 3)}\nUsa --camera-order para fijarlas a mano.")
        self.slot_of_view = {VIEWS[v]: s for s, v in enumerate(perm)}
        self.margin = margin
        return True

    def identify(self, jpegs: list[bytes]) -> list[tuple[str, float]]:
        """Para un nº cualquiera de imágenes: (vista más parecida, puntuación) de cada una. Solo diagnóstico."""
        if not self.bank:
            return [("?", 0.0)] * len(jpegs)
        out = []
        for j in jpegs:
            f = _feat(j)
            sc = [float(np.sort(b @ f / f.size)[-self.k:].mean()) for b in self.bank]
            out.append((VIEWS[int(np.argmax(sc))], max(sc)))
        return out

    def by_view(self, jpegs: list[bytes]) -> dict[str, bytes]:
        assert self.slot_of_view is not None
        return {v: jpegs[s] for v, s in self.slot_of_view.items()}

    def describe(self) -> str:
        assert self.slot_of_view is not None
        order = sorted(self.slot_of_view.items(), key=lambda kv: kv[1])
        return ", ".join(f"images[{s}]={v}" for v, s in order)

    def save_preview(self, jpegs: list[bytes], path: Path) -> None:
        """Guarda una lámina side|gripper|table para comprobar a ojo la asignación."""
        bv = self.by_view(jpegs)
        ims = [Image.open(io.BytesIO(bv[v])).convert("RGB").resize((224, 224)) for v in VIEWS]
        sheet = Image.new("RGB", (224 * 3, 224))
        for i, im in enumerate(ims):
            sheet.paste(im, (224 * i, 0))
        sheet.save(path)


def decode_rgb(jpeg: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.uint8)
