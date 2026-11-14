# Leukerbad Hackathon 2026 — Rehab & Elderly Care

Project for the **Leukerbad Robotics Hackathon 2026** (Nov 14–16, 2026, Leukerbad, Switzerland), **Elderly Care & Rehabilitation** track.

## Idea

Use a **LeRobot SO-101** robotic arm (open-source, 6 DOF, supported by Hugging Face LeRobot) to assist **rehabilitation of patients recovering from hand surgery**: guided reach/manipulation exercises, progress logging, and **remote supervision by a doctor/therapist**.

## Planned components

- **Hardware**: LeRobot SO-101 arm (leader/follower, Feetech STS3215 servos).
- **Software**: `lerobot` (Hugging Face) for control/calibration; camera for session tracking; remote interface so the clinician can supervise and adjust difficulty.
- **Demo**: a functional exercise session (reach/grasp of light objects) with metrics on completed tasks, assistance needed, and variability, visible to the remote supervisor.

## Status

Repo just created — structure and code in progress during the hackathon.

See [AGENTS.md](./AGENTS.md) for the git/GitHub workflow.
