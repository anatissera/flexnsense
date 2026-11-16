"""The pull arm as a training task: hold the scale's ring, and give way when the pull exceeds the limit.

One ElRobot per environment. The scale and the finger are modelled as in sim/03_yield_loop.py: a spring from the point
between the jaws to a finger point that moves away and comes back. The simulator knows the true force; the policy gets
a noisy copy, one step late (in the real rig that reading comes from the camera on the scale).

What the policy sees (29 numbers): the seven arm joints (position relative to the start posture, speed), how far each
driven joint is from its target, the force reading now and one step ago (in units of the limit), where the gripper is
relative to its starting place (along the scale, across it, up), and its last action.
What it does (5 numbers): changes to the targets of motors 1, 2, 3, 4 and 6.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, quat_from_angle_axis, sample_uniform

from .pull_env_cfg import PullEnvCfg


class PullEnv(DirectRLEnv):
    cfg: PullEnvCfg

    def __init__(self, cfg: PullEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        dev, n = self.device, self.num_envs
        self.act_ids = torch.tensor(self.robot.find_joints("rev_motor_0[12346]", preserve_order=True)[0], device=dev)
        self.arm_ids = torch.tensor(self.robot.find_joints("rev_motor_0[1-7]", preserve_order=True)[0], device=dev)
        self.ee = self.robot.find_bodies("Gripper_Base_v1_1")[0][0]
        self.limits = self.robot.data.joint_pos_limits.torch[0]
        self.default_q = self.robot.data.default_joint_pos.torch.clone()
        self.q_tgt = self.default_q.clone()
        self.q_prev = self.default_q.clone()
        self.q_cmd = self.default_q.clone()  # what is actually sent to the servos (the target, possibly lagged)
        self.follow = torch.ones(n, device=dev)
        self.point = torch.tensor([0.0, 1.0, 0.0], device=dev)  # the gripper points along its own +y
        self.up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        z = lambda *s: torch.zeros(n, *s, device=dev)
        self.home, self.u, self.side_dir = z(3), z(3), z(3)
        self.need_home = torch.ones(n, dtype=torch.bool, device=dev)
        self.k, self.amp, self.ramp, self.hold, self.delay = z(), z(), z(), z(), z()
        self.force, self.force_obs, self.force_prev = z(), z(), z()
        self.last_action, self.action = z(5), z(5)
        self.max_force, self.over_time, self.give = z(), z(), z()
        if self.cfg.bench:
            self.hand, self.finger, self.tube = self.scene["hand"], self.scene["finger"], self.scene["tube"]

    # ------------------------------------------------------------------ helpers
    def _tip(self):
        pose = self.robot.data.body_pose_w.torch[:, self.ee]
        pointing = matrix_from_quat(pose[:, 3:7]) @ self.point
        return pose[:, 0:3] + pointing * self.cfg.tip, pointing

    def _pull(self):
        """How far the finger has moved away [m]: wait, build up, hold, come back."""
        t = self.episode_length_buf * self.step_dt - self.delay
        up_ = torch.clamp(t / self.ramp, 0.0, 1.0)
        down = torch.clamp((t - self.ramp - self.hold) / self.ramp, 0.0, 1.0)
        smooth = lambda p: p * p * (3 - 2 * p)
        return self.amp * (smooth(up_) - smooth(down))

    def _set_home(self):
        ids = self.need_home.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        tip, _ = self._tip()
        self.home[ids] = tip[ids]
        base = self.robot.data.root_pose_w.torch[ids, 0:3]
        line = tip[ids] - base
        line[:, 2] = 0.0
        line = line / line.norm(dim=-1, keepdim=True)
        yaw = sample_uniform(-self.cfg.yaw_range, self.cfg.yaw_range, (len(ids),), self.device)
        c, s = torch.cos(yaw), torch.sin(yaw)
        self.u[ids] = torch.stack([c * line[:, 0] - s * line[:, 1], s * line[:, 0] + c * line[:, 1], torch.zeros_like(c)], dim=-1)
        self.side_dir[ids] = torch.linalg.cross(self.up[ids], self.u[ids])
        self.need_home[ids] = False
        if self.cfg.bench:  # the hand goes where this arm's scale ends: index finger at the loop, palm beyond it
            stand = self.home[ids] + self.u[ids] * self.cfg.rest
            stand[:, 2] = self.scene.env_origins[ids, 2]
            yaw = torch.atan2(self.u[ids, 1], self.u[ids, 0]) - torch.pi / 2
            quat = quat_from_angle_axis(yaw, self.up[ids])
            self.hand.write_root_pose_to_sim_index(root_pose=torch.cat([stand, quat], dim=-1), env_ids=ids)

    def _place_scale(self, tip: torch.Tensor, finger: torch.Tensor):
        """Put the pulling finger and the scale's tube where they are this instant (they are shown, not simulated:
        the scale's pull on the arm is the spring force, which is what a light tube between two points transmits)."""
        n = self.num_envs
        ident = torch.zeros(n, 4, device=self.device)
        ident[:, 3] = 1.0
        f_pos = finger.clone()
        f_pos[:, 2] = self.scene.env_origins[:, 2] + 0.04
        self.finger.write_root_pose_to_sim_index(root_pose=torch.cat([f_pos, ident], dim=-1))
        d = finger - tip
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        centre = finger - d * (0.044 + 0.165 / 2)  # the cap and the wire loop take the last 44 mm
        axis = torch.linalg.cross(self.up, d)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        quat = quat_from_angle_axis(torch.acos(d[:, 2].clamp(-1.0, 1.0)), axis)
        self.tube.write_root_pose_to_sim_index(root_pose=torch.cat([centre, quat], dim=-1))

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_action = self.action.clone()
        self.action = actions.clamp(-1.0, 1.0)
        q = self.q_tgt[:, self.act_ids] + self.cfg.action_scale * self.action
        self.q_tgt[:, self.act_ids] = torch.clamp(q, self.limits[self.act_ids, 0], self.limits[self.act_ids, 1])
        self.q_cmd = self.q_cmd + self.follow.unsqueeze(-1) * (self.q_tgt - self.q_cmd)
        # the scale: a spring from the point between the jaws to the finger
        tip, _ = self._tip()
        finger = self.home + self.u * (self.cfg.rest + self._pull().unsqueeze(-1))
        d = finger - tip
        dist = d.norm(dim=-1)
        self.force = self.k * torch.clamp(dist - self.cfg.rest, min=0.0)
        # Right after a reset the simulator has not yet moved the arm to its new posture, so the starting place is
        # not known for one step: no spring force until then (see _get_dones).
        self.force = torch.where(self.need_home, torch.zeros_like(self.force), self.force)
        dist = torch.clamp(dist, min=1e-6)
        self._wrench = (d / dist.unsqueeze(-1) * self.force.unsqueeze(-1)).unsqueeze(1)
        self._wrench_at = tip.unsqueeze(1)
        if self.cfg.bench:
            self._place_scale(tip, torch.where(self.need_home.unsqueeze(-1), tip + self.u * self.cfg.rest, finger))

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(target=self.q_cmd)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._wrench, positions=self._wrench_at, body_ids=[self.ee], is_global=True
        )

    def _rel(self):
        tip, pointing = self._tip()
        off = torch.where(self.need_home.unsqueeze(-1), torch.zeros_like(tip), tip - self.home)
        along = (off * self.u).sum(-1)
        side = (off * self.side_dir).sum(-1)
        return along, side, off[:, 2], pointing

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos.torch
        qd = self.robot.data.joint_vel.torch
        if self.cfg.finite_diff_velocity:  # what a real loop can compute from two consecutive frames
            qd = (q - self.q_prev) / self.step_dt
            self.q_prev = q.clone()
        along, side, height, _ = self._rel()
        self.force_prev = self.force_obs.clone()
        reading = self.force + self.cfg.force_noise * torch.randn_like(self.force)
        if self.cfg.force_step > 0:  # the camera reads the scale in steps
            reading = torch.round(reading / self.cfg.force_step) * self.cfg.force_step
        if self.cfg.force_dropout > 0:  # a lost frame: the last reading stays
            lost = torch.rand_like(reading) < self.cfg.force_dropout
            reading = torch.where(lost, self.force_obs, reading)
        self.force_obs = reading
        obs = torch.cat(
            [
                (q - self.default_q)[:, self.arm_ids],
                0.1 * qd[:, self.arm_ids],
                (self.q_tgt - q)[:, self.act_ids],
                (self.force_prev / self.cfg.cap).unsqueeze(-1),
                ((self.force_obs - self.force_prev) / self.cfg.cap).unsqueeze(-1),
                10.0 * torch.stack([along, side, height], dim=-1),
                self.action,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        along, side, height, pointing = self._rel()
        aim = cfg.reward_cap_share * cfg.cap
        over = torch.clamp(self.force - aim, min=0.0)
        low = (self.force < 0.8 * aim).float()
        aim_err = 1.0 - (pointing * self.u).sum(-1)
        rate = (self.action - self.last_action).square().sum(-1)
        rew = (
            cfg.rew_alive
            + cfg.rew_over * (over + over.square())
            + cfg.rew_hold * low * along.abs()
            + cfg.rew_side * (side.abs() + height.abs())
            + cfg.rew_aim * aim_err
            + cfg.rew_rate * rate
        )
        self.max_force = torch.maximum(self.max_force, self.force)
        self.over_time += (self.force > 1.1 * cfg.cap).float() * self.step_dt
        self.give = torch.maximum(self.give, along)
        return rew * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._set_home()  # the first moment after a reset at which the arm's new posture can be read
        tip, _ = self._tip()
        lost = (tip - self.home).norm(dim=-1) > self.cfg.max_drift
        return lost, self.episode_length_buf >= self.max_episode_length

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        log = self.extras.setdefault("log", {})
        log["Metrics/peak_force_N"] = self.max_force[env_ids].mean()
        log["Metrics/seconds_above_110pct_limit"] = self.over_time[env_ids].mean()
        log["Metrics/give_m"] = self.give[env_ids].mean()
        log["Metrics/peak_force_if_held_N"] = (self.k[env_ids] * self.amp[env_ids]).mean()
        super()._reset_idx(env_ids)
        n, dev, cfg = len(env_ids), self.device, self.cfg
        q = self.default_q[env_ids].clone()
        q[:, self.act_ids[1:]] += sample_uniform(-cfg.joint_noise, cfg.joint_noise, (n, 4), dev)
        q = torch.clamp(q, self.limits[:, 0], self.limits[:, 1])
        root = self.robot.data.default_root_pose.torch[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.q_tgt[env_ids] = q
        self.q_prev[env_ids] = q
        self.q_cmd[env_ids] = q
        if cfg.follow_range is not None:
            self.follow[env_ids] = sample_uniform(cfg.follow_range[0], cfg.follow_range[1], (n,), dev)
        if cfg.stiffness_range is not None:  # a different servo for every arm and episode
            k = sample_uniform(cfg.stiffness_range[0], cfg.stiffness_range[1], (n, 1), dev).expand(n, len(self.arm_ids))
            d = sample_uniform(cfg.damping_range[0], cfg.damping_range[1], (n, 1), dev).expand(n, len(self.arm_ids))
            self.robot.write_joint_stiffness_to_sim_index(stiffness=k.contiguous(), joint_ids=self.arm_ids, env_ids=env_ids)
            self.robot.write_joint_damping_to_sim_index(damping=d.contiguous(), joint_ids=self.arm_ids, env_ids=env_ids)
        self.need_home[env_ids] = True
        rnd = lambda r: sample_uniform(r[0], r[1], (n,), dev)
        self.k[env_ids] = cfg.spring * rnd(cfg.spring_range)
        self.amp[env_ids], self.ramp[env_ids] = rnd(cfg.pull_range), rnd(cfg.ramp_range)
        self.hold[env_ids], self.delay[env_ids] = rnd(cfg.hold_range), rnd(cfg.delay_range)
        for buf in (self.force, self.force_obs, self.force_prev, self.max_force, self.over_time, self.give):
            buf[env_ids] = 0.0
        self.action[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
