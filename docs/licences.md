# Licences

This repository is MIT ([LICENSE](../LICENSE)). The pieces it builds on are not all as
permissive, and two of them matter for anything beyond research.

| Component | Licence | What it means here |
|---|---|---|
| NormaCore Station and the ElRobot URDF | MIT | no constraint |
| LeRobot | Apache 2.0 | no constraint |
| MediaPipe, OpenCV, NumPy | Apache 2.0 / BSD | no constraint |
| MuJoCo (dashboard, WebAssembly) | Apache 2.0 | no constraint |
| NVIDIA Isaac Sim and Isaac Lab | NVIDIA licence / BSD-3 | free for research; redistribution of Omniverse components has its own terms |
| **WiLoR** | **CC BY-NC-ND 4.0** | **non-commercial, no derivatives.** Used for hand keypoints in arm 2's training data and as the dashboard's precision tracker |
| **MANO** (WiLoR depends on it) | own licence, registration required | separate terms, must be accepted directly |

## The one that would block a product

**A commercial product could not ship WiLoR or MANO as they stand.** Non-commercial is the
binding word, and "no derivatives" rules out fine-tuning the model on our own episodes.

MediaPipe is the way out, and it is already the default: it is Apache 2.0, it runs on CPU, and
the dashboard uses it unless WiLoR is explicitly selected. What WiLoR buys is precision in the
keypoints written into arm 2's training data, so the question to settle before any of this
leaves research is whether a MediaPipe-only dataset trains a good enough hook mover.

Nothing in `arm1/`, the part that touches the patient, depends on WiLoR at all.
