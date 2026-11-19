"""Shared helpers for every script in this repo (robot arms, force calibration, gauge reader).

    from arm1.lib.camera import open_source, list_cameras    # cameras by number or name ("iphone"), warm-up, frames
    from arm1.lib.env import env_get                          # values from the environment or .env (CAMERA_INDEX, ROBOT_PORT)
    from arm1.lib.filters import ReadingFilter                # 1D Kalman filter: smooth a noisy signal + "steady" flag
    from arm1.lib.geometry import Axis                        # a line in an image: project points, distances along it

Nothing here knows about the PESOLA scale; that lives in gauge_reader/.
"""
