"""WiLoR as a HandFrame source: the same shape MediaPipe emits, plus the MANO block.

WiLoR is the precision-capture path, not the live one. On a 4 GB RTX 3050 in fp16 it runs at
about 11 FPS on 640x480 with two hands in frame, against MediaPipe's video-rate in the browser,
and it needs PyTorch, CUDA and a 2.5 GB checkpoint. Use it over recorded clips or at a low rate
when the MANO rotations are what you are after; see docs/hand-tracking.md for the measurements.

What it adds over MediaPipe is ``HandFrame.mano``: per-joint rotations with all three degrees of
freedom, rather than one flexion angle per joint. Those can fill the MyoHand model's abduction
joints, which a bend angle between two bones cannot reach.

Two things make the two sources line up with no conversion layer:

- WiLoR's MANO wrapper remaps its joints to OpenPose order (``mano_to_openpose`` in
  ``wilor/models/mano_wrapper.py``), and OpenPose's 21-point hand layout is the same one
  MediaPipe uses. So ``pred_keypoints_3d`` indexes exactly like MediaPipe's landmarks and
  :func:`hand_tracking.pose.landmarks_to_pose` applies to it unchanged.
- The angles therefore come out of the same function for both sources.

torch, wilor and ultralytics are imported lazily inside :class:`WilorTracker`, so the pure
geometry in this module stays importable (and testable) on a machine with none of them.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from arm1.hand_tracking.frame import HandFrame, ManoParams
from arm1.hand_tracking.pose import Point, landmarks_to_pose, tip_wrist_mm

#: Default locations inside a WiLoR checkout, relative to its root.
CHECKPOINT = "pretrained_models/wilor_final.ckpt"
MODEL_CFG = "pretrained_models/model_config.yaml"
DETECTOR = "pretrained_models/detector.pt"


def rotation_matrix_to_axis_angle(matrix: Any) -> tuple[float, float, float]:
    """A 3x3 rotation matrix to an axis-angle vector in radians (its length is the angle).

    MANO's own parameterisation is axis-angle; WiLoR predicts rotation matrices instead, so a
    HandFrame converts back on the way out, which is both more compact and what a consumer
    expecting MANO parameters will want.
    """
    R = np.asarray(matrix, dtype=float).reshape(3, 3)
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))
    if angle < 1e-6:
        return (0.0, 0.0, 0.0)
    sin = math.sin(angle)
    if sin > 1e-6:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * sin)
    else:
        # Half a turn: the skew part vanishes, so take the axis from the symmetric part instead.
        # For a rotation by pi about unit a, (R + I)/2 is the outer product a a^T.
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        magnitude = math.sqrt(max(float(A[k, k]), 0.0))
        axis = A[k] / magnitude if magnitude > 1e-9 else np.array([1.0, 0.0, 0.0])
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    v = axis * angle
    return (float(v[0]), float(v[1]), float(v[2]))


def project_to_normalized(
    points_3d: Any, cam_t: Any, focal_length: float, width: int, height: int
) -> list[Point]:
    """Camera-space points to normalized image coordinates, matching WiLoR's demo projection.

    x and y come back as fractions of the image width and height, which is what a HandFrame's
    ``landmarks`` are. z is the camera-space depth in metres, kept for ordering, and is the one
    component whose meaning is source specific.
    """
    pts = np.asarray(points_3d, dtype=float) + np.asarray(cam_t, dtype=float)
    z = pts[:, 2:3]
    # A joint at the pinhole has no projection; nudge rather than divide by zero.
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    projected = pts / safe_z
    x_px = focal_length * projected[:, 0] + width / 2.0
    y_px = focal_length * projected[:, 1] + height / 2.0
    return [
        (float(x / width), float(y / height), float(depth))
        for x, y, depth in zip(x_px, y_px, z[:, 0])
    ]


def build_frame(
    *,
    t_ms: float,
    is_right: bool,
    keypoints_3d: Any,
    cam_t: Any,
    focal_length: float,
    width: int,
    height: int,
    hand_pose: Any,
    global_orient: Any,
    betas: Any,
) -> HandFrame:
    """Assemble one HandFrame from the arrays a WiLoR forward pass produces.

    ``keypoints_3d`` is expected already un-mirrored (WiLoR feeds left hands through the network
    flipped, so the demo multiplies x by ``2 * is_right - 1`` on the way out); :meth:`WilorTracker.track`
    does that before calling this.
    """
    world = [(float(p[0]), float(p[1]), float(p[2])) for p in np.asarray(keypoints_3d, dtype=float)]
    landmarks = project_to_normalized(keypoints_3d, cam_t, focal_length, width, height)
    mano = ManoParams(
        hand_pose=[rotation_matrix_to_axis_angle(m) for m in np.asarray(hand_pose, dtype=float).reshape(-1, 3, 3)],
        global_orient=rotation_matrix_to_axis_angle(np.asarray(global_orient, dtype=float).reshape(3, 3)),
        betas=[float(b) for b in np.asarray(betas, dtype=float).reshape(-1)],
    )
    return HandFrame(
        source="wilor",
        t_ms=t_ms,
        handedness="right" if is_right else "left",
        landmarks=landmarks,
        world=world,
        angles=landmarks_to_pose(world),
        tip_wrist_mm=tip_wrist_mm(world),
        mano=mano,
    )


class WilorTracker:
    """Runs WiLoR over frames and yields HandFrames. Needs a CUDA GPU and a WiLoR checkout.

    The heavy imports happen in the constructor, not at module import, so that this file can be
    read and its geometry tested without torch installed.
    """

    def __init__(
        self,
        wilor_root: str,
        *,
        fast: bool = True,
        conf: float = 0.3,
        rescale_factor: float = 2.0,
        device: str | None = None,
    ) -> None:
        import os
        import sys

        import torch
        from ultralytics import YOLO

        # wilor resolves its MANO data through paths relative to its own root.
        wilor_root = os.path.abspath(wilor_root)
        if wilor_root not in sys.path:
            sys.path.insert(0, wilor_root)
        cwd = os.getcwd()
        os.chdir(wilor_root)
        try:
            from wilor.models import load_wilor

            model, model_cfg = load_wilor(checkpoint_path=CHECKPOINT, cfg_path=MODEL_CFG)
            detector = YOLO(DETECTOR)
        finally:
            os.chdir(cwd)

        if fast:
            torch.set_float32_matmul_precision("high")
            model = model.half()
            model.backbone = torch.compile(model.backbone)
            model.backbone.skip_blocks = True

        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.detector = detector.to(self.device)
        self.model_cfg = model_cfg
        self.fast = fast
        self.conf = conf
        self.rescale_factor = rescale_factor

    def track(self, frame_bgr: Any, t_ms: float) -> list[HandFrame]:
        """One frame in, one HandFrame per hand found out (empty list when none are)."""
        import torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full

        detections = self.detector(frame_bgr, conf=self.conf, verbose=False)[0]
        boxes, rights = [], []
        for det in detections:
            box = det.boxes.data.cpu().detach().squeeze().numpy()
            rights.append(det.boxes.cls.cpu().detach().squeeze().item())
            boxes.append(box[:4].tolist())
        if not boxes:
            return []

        height, width = frame_bgr.shape[:2]
        dataset = ViTDetDataset(
            self.model_cfg, frame_bgr, np.stack(boxes), np.stack(rights),
            rescale_factor=self.rescale_factor, fp16=self.fast,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        frames: list[HandFrame] = []
        for batch in loader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            # Left hands go through the network mirrored, so the camera x and the predicted x
            # both have to be flipped back. This matches WiLoR's own demo.py.
            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            focal = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            cam_t_full = cam_crop_to_full(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(), img_size, focal
            ).detach().cpu().numpy()

            mano = out["pred_mano_params"]
            for n in range(batch["img"].shape[0]):
                is_right = bool(batch["right"][n].cpu().numpy())
                joints = out["pred_keypoints_3d"][n].detach().cpu().float().numpy().copy()
                joints[:, 0] = (2 * int(is_right) - 1) * joints[:, 0]
                frames.append(build_frame(
                    t_ms=t_ms,
                    is_right=is_right,
                    keypoints_3d=joints,
                    cam_t=cam_t_full[n],
                    focal_length=float(focal),
                    width=width,
                    height=height,
                    hand_pose=mano["hand_pose"][n].detach().cpu().float().numpy(),
                    global_orient=mano["global_orient"][n].detach().cpu().float().numpy(),
                    betas=mano["betas"][n].detach().cpu().float().numpy(),
                ))
        return frames
