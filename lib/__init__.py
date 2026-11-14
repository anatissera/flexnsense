"""Shared helpers for every script in this repo (robot arms, force calibration, gauge reader).

    from lib.camera import open_source, list_cameras    # cameras by number or name ("iphone"), warm-up, frames
    from lib.env import env_get                          # values from the environment or .env (CAMERA_INDEX, ROBOT_PORT)
    from lib.filters import ReadingFilter                # 1D Kalman filter: smooth a noisy signal + "steady" flag
    from lib.geometry import Axis                        # a line in an image: project points, distances along it

Nothing here knows about the PESOLA scale; that lives in gauge_reader/.
"""
