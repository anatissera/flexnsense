"""Gauge reader (roadmap item 2): read the PESOLA 600 g spring scale through a camera and output grams.

    python -m gauge_reader cameras | calibrate | run | evaluate     # command line (cli.py)
    python -m gauge_reader.launcher                                 # control panel

Modules: stickers.py (find the two red dots + marker), detector.py (red mask; fixed-mode marker),
calibration.py (position -> grams), overlay.py (drawing), cli.py, launcher.py, synthetic/ (test videos).
Generic pieces (camera, filters, geometry, env) live in the shared lib/ package.
"""
