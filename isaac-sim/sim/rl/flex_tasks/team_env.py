"""Both follower arms on one wooden table, learning their two jobs at the same time, each with its own policy.

Agents:
  "pull"  holds the scale's ring; gives way when the force passes the limit; holds still while the scale is moved.
  "move"  sleeps while a finger pulls; when the pull ends it wakes up, takes the scale near its loop, lifts the loop
          over the finger, sets it on the other finger, lets go and goes back to sleep.
Each has its own network, its own observations and its own reward. They share the outcome: both are rewarded when
the loop is placed, both are penalised when the move fails, and the episode ends for both.

One episode (20 s, 10 decisions a second) starts in one of two ways, so that the arms learn more than one situation:
  * the loop is on a finger: that finger pulls, the loop is moved to the other finger, that one pulls;
  * the loop lies on the table beside the hand: the move arm picks it up and puts it on a finger, which then pulls.

The force the arms see is read the way the overhead camera reads the scale: three points on the tube, two fixed and
one that slides with the force. The reading is the sliding point's place between the fixed ones in whole camera
pixels, with noise, and it is missing while the move arm's gripper covers the sliding point.

Taking and placing follow rules, not contact physics (see duo_env.py). The geometry, timing and rules are shared with
that task; this one differs in that the pull arm learns too.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import matrix_from_quat, sample_uniform

from .duo_env import DuoEnv
from .duo_env_cfg import MOVE_YAW
from .team_env_cfg import TeamEnvCfg

PULL_START = [0.0, 0.04, 1.057, 0.072, 0.0, -1.169, 0.0]


class TeamEnv(DirectMARLEnv):
    cfg: TeamEnvCfg

    # geometry, timing and the take/place rules are the same code as the two-arm task with a fixed pull policy
    _tips, _time, _phase, _pull_now = DuoEnv._tips, DuoEnv._time, DuoEnv._phase, DuoEnv._pull_now
    _finger, _on_finger, _place_tol = DuoEnv._finger, DuoEnv._on_finger, DuoEnv._place_tol
    _grasp_point, _set_home, _rules, _sleep_error = DuoEnv._grasp_point, DuoEnv._set_home, DuoEnv._rules, DuoEnv._sleep_error
    _move_reward, _move_obs = DuoEnv._get_rewards, DuoEnv._get_observations

    def __init__(self, cfg: TeamEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.A, self.B = self.scene["pull_arm"], self.scene["move_arm"]
        dev, n = self.device, self.num_envs
        ids = lambda expr: torch.tensor(self.A.find_joints(expr, preserve_order=True)[0], device=dev)
        self.act_ids, self.arm_ids, self.jaw_ids = ids("rev_motor_0[12346]"), ids("rev_motor_0[1-7]"), ids("rev_motor_08_[12]")
        self.ee = self.A.find_bodies("Gripper_Base_v1_1")[0][0]
        self.limits = self.A.data.joint_pos_limits.torch[0]
        self.A_default = self.A.data.default_joint_pos.torch.clone()
        self.B_default = self.B.data.default_joint_pos.torch.clone()
        self.pull_start = torch.tensor(PULL_START, device=dev)
        self.point = torch.tensor([0.0, 1.0, 0.0], device=dev)
        self.up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        # working tilt of the move arm's gripper: 30 degrees forward of straight down, forward being the way it faces
        fwd = (-math.sin(MOVE_YAW), math.cos(MOVE_YAW))
        s30, c30 = math.sin(math.radians(30)), math.cos(math.radians(30))
        self.work_dir = torch.tensor([fwd[0] * s30, fwd[1] * s30, -c30], device=dev)
        z = lambda *s: torch.zeros(n, *s, device=dev)
        b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
        self.qA_tgt, self.qA_prev = self.A_default.clone(), self.A_default.clone()
        self.actA, self.last_actA = z(5), z(5)
        self.force, self.force_obs, self.force_prev = z(), z(), z()
        self.seen = torch.ones(n, dtype=torch.bool, device=dev)  # the camera sees the sliding point
        self.home, self.u0, self.hdir = z(3), z(3), z(3)
        self.qB_tgt, self.qB_prev = self.B_default.clone(), self.B_default.clone()
        self.grip, self.action, self.last_action = z(), z(6), z(6)
        self.stand = z(2, 3)
        self.src = torch.zeros(n, dtype=torch.long, device=dev)
        self.loop, self.take_off = z(3), z(3)
        self.held, self.placed, self.dropped, self.snagged, self.late = b(), b(), b(), b(), b()
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.free = b()  # the loop lies on the table, on no finger (one of the two ways an episode starts)
        self.k, self.amp, self.ramp, self.hold, self.delay = z(), z(2), z(2), z(2), z(2)
        self.t1, self.t2 = z(), z()
        self.ev_take, self.ev_place, self.ev_drop, self.ev_snag, self.ev_late = z(), z(), z(), z(), z()
        self.peak1, self.peak2, self.t_place, self.lifted, self.over_time = z(), z(), z(), z(), z()
        self.qA_cmd, self.qB_cmd = self.A_default.clone(), self.B_default.clone()  # what is sent to the servos (lagged goals)
        self.followA, self.followB = torch.ones(n, device=dev), torch.ones(n, device=dev)
        self.tipB_prev, self.top_speed = z(3), z()

    # ------------------------------------------------------------------ the scale's three points, seen from above
    def _indicator(self, tip_a: torch.Tensor):
        """Where the sliding point is on the tube: it starts by the loop end and runs towards the ring with the force."""
        d = self.loop - tip_a
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        travel = self.cfg.scale_travel * torch.clamp(self.force / self.cfg.full_scale, max=1.0)
        return self.loop - d * (0.080 + travel).unsqueeze(-1)

    def _read_scale(self, tip_a: torch.Tensor, tip_b: torch.Tensor):
        c = self.cfg
        px_n = c.full_scale / (c.scale_travel * c.camera_px_per_m)  # newtons per camera pixel
        reading = self.force + c.force_noise * torch.randn_like(self.force)
        reading = torch.clamp(torch.round(reading / px_n) * px_n, min=0.0)
        covered = (tip_b[:, 0:2] - self._indicator(tip_a)[:, 0:2]).norm(dim=-1) < c.cover_radius
        self.seen = ~covered
        self.force_prev = self.force_obs.clone()
        self.force_obs = torch.where(covered, self.force_obs, reading)  # no new reading: the last one stays

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        c = self.cfg
        tip_a, _, _ = self._tips()
        live = ~self.fresh
        stand, u = self._finger(self._on_finger())
        # the pull arm: its policy's action
        self.last_actA = self.actA.clone()
        self.actA = torch.where(live.unsqueeze(-1), actions["pull"].clamp(-1.0, 1.0), torch.zeros_like(self.actA))
        qa_now = self.A.data.joint_pos.torch[:, self.act_ids]
        q = self.qA_tgt[:, self.act_ids] + c.pull_action_scale * self.actA
        q = torch.maximum(torch.minimum(q, qa_now + c.pull_max_lead), qa_now - c.pull_max_lead)  # never far ahead of the arm
        self.qA_tgt[:, self.act_ids] = torch.clamp(q, self.limits[self.act_ids, 0], self.limits[self.act_ids, 1])
        # the scale: a spring from the pull arm's jaws to the finger its loop is on; no pull while it is carried
        finger = stand + u * self._pull_now().unsqueeze(-1)
        on_a_finger = live & ~self.held & ~self.free
        self.loop = torch.where(on_a_finger.unsqueeze(-1), finger, self.loop)
        d = self.loop - tip_a
        dist = d.norm(dim=-1).clamp(min=1e-6)
        force = self.k * torch.clamp(dist - c.rest, min=0.0)
        self.force = torch.where(on_a_finger, force, torch.zeros_like(force))
        self._wrench = (d / dist.unsqueeze(-1) * self.force.unsqueeze(-1)).unsqueeze(1)
        self._wrench_at = (tip_a + self.scene.env_origins).unsqueeze(1)
        # the move arm: its policy's action
        self.last_action = self.action.clone()
        self.action = actions["move"].clamp(-1.0, 1.0)
        qb_now = self.B.data.joint_pos.torch[:, self.act_ids]
        q = self.qB_tgt[:, self.act_ids] + c.action_scale * self.action[:, :5]
        q = torch.maximum(torch.minimum(q, qb_now + c.move_max_lead), qb_now - c.move_max_lead)
        self.qB_tgt[:, self.act_ids] = torch.clamp(q, self.limits[self.act_ids, 0], self.limits[self.act_ids, 1])
        self.grip = torch.clamp(self.grip + c.grip_rate * self.action[:, 5], 0.0, 1.0)
        lo, hi = self.limits[self.jaw_ids, 0], self.limits[self.jaw_ids, 1]
        shut = torch.where(self.held, self.grip.clamp(max=0.62), self.grip)
        self.qB_tgt[:, self.jaw_ids] = torch.stack([hi[0] + (lo[0] - hi[0]) * shut, lo[1] + (hi[1] - lo[1]) * shut], dim=-1)
        # the servos follow their goals with a lag that differs from table to table
        self.qA_cmd = self.qA_cmd + self.followA.unsqueeze(-1) * (self.qA_tgt - self.qA_cmd)
        self.qB_cmd = self.qB_cmd + self.followB.unsqueeze(-1) * (self.qB_tgt - self.qB_cmd)
        self.qB_cmd[:, self.jaw_ids] = self.qB_tgt[:, self.jaw_ids]

    def _apply_action(self) -> None:
        self.A.set_joint_position_target_index(target=self.qA_cmd)
        self.B.set_joint_position_target_index(target=self.qB_cmd)
        self.A.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._wrench, positions=self._wrench_at, body_ids=[self.ee], is_global=True
        )

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._rules()
        failed = self.dropped | self.snagged | self.late
        over = self.episode_length_buf >= self.max_episode_length - 1
        return {a: failed for a in self.cfg.possible_agents}, {a: over for a in self.cfg.possible_agents}

    def _pull_frame(self, tip_a: torch.Tensor):
        _, u = self._finger(self._on_finger())
        side = torch.linalg.cross(self.up.expand_as(u), u)
        off = torch.where(self.fresh.unsqueeze(-1), torch.zeros_like(tip_a), tip_a - self.home)
        return u, (off * u).sum(-1), (off * side).sum(-1), off[:, 2]

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        c, dt = self.cfg, self.step_dt
        r_move = self._move_reward()
        tip_a, tip_b = self._now[0], self._now[1]
        speed = torch.where(self.fresh | (self.episode_length_buf < 3), torch.zeros_like(self.force),
                            (tip_b - self.tipB_prev).norm(dim=-1) / dt)
        self.tipB_prev = tip_b.clone()
        self.top_speed = torch.maximum(self.top_speed, speed)
        r_move = r_move + c.rew_speed * torch.clamp(speed - c.max_tip_speed, min=0.0) * dt
        lead_b = (self.qB_tgt - self.B.data.joint_pos.torch)[:, self.act_ids].abs().mean(dim=-1)
        r_move = r_move + c.rew_m_lead * lead_b * dt
        u, along, side, height = self._pull_frame(tip_a)
        pa = self.A.data.body_pose_w.torch[:, self.ee]
        pointing_a = matrix_from_quat(pa[:, 3:7]) @ self.point
        aim_n = c.reward_cap_share * c.cap
        over = torch.clamp(self.force - aim_n, min=0.0)
        low = (self.force < 0.8 * aim_n).float()
        rate = (self.actA - self.last_actA).square().sum(-1)
        r_pull = (
            c.rew_p_alive
            + c.rew_p_over * (over + over.square())
            + c.rew_p_hold * low * along.abs()
            + c.rew_p_side * (side.abs() + height.abs())
            + c.rew_p_aim * (1.0 - (pointing_a * self.u0).sum(-1))
            + c.rew_p_rate * rate
            + c.rew_p_lead * (self.qA_tgt - self.A.data.joint_pos.torch)[:, self.act_ids].abs().mean(dim=-1)
        ) * dt
        r_pull = r_pull + c.rew_team_place_once * self.ev_place + c.rew_team_fail_once * (self.ev_drop + self.ev_snag + self.ev_late)
        self.over_time += (self.force > 1.1 * c.cap).float() * dt
        return {"pull": r_pull, "move": r_move}

    def _get_observations(self) -> dict[str, torch.Tensor]:
        c = self.cfg
        tip_a, tip_b, _ = self._tips()
        self._read_scale(tip_a, tip_b)
        ph = self._phase()
        qa = self.A.data.joint_pos.torch
        u, along, side, height = self._pull_frame(tip_a)
        obs_pull = torch.cat(
            [
                qa[:, self.arm_ids] - self.pull_start,
                0.1 * (qa - self.qA_prev)[:, self.arm_ids] / self.step_dt,
                (self.qA_tgt - qa)[:, self.act_ids],
                (self.force_prev / c.cap).unsqueeze(-1),
                ((self.force_obs - self.force_prev) / c.cap).unsqueeze(-1),
                10.0 * torch.stack([along, side, height], dim=-1),
                self.actA,
                torch.nn.functional.one_hot(ph, 3).float(),
                self.held.float().unsqueeze(-1),
                self.seen.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        self.qA_prev = qa.clone()
        return {"pull": obs_pull, "move": self._move_obs()["policy"]}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.A._ALL_INDICES
        log = self.extras.setdefault("log", {})
        f = lambda v: v[env_ids].float().mean()
        log["Metrics/placed"], log["Metrics/dropped"] = f(self.placed), f(self.dropped)
        log["Metrics/hit_a_finger"], log["Metrics/too_late"] = f(self.snagged), f(self.late)
        log["Metrics/taken"] = f(self.held | self.placed | self.dropped | self.snagged)
        if hasattr(self, "_was_free"):
            wf = self._was_free[env_ids]
            log["Metrics/placed_from_table"] = (self.placed[env_ids] & wf).float().sum() / wf.float().sum().clamp(min=1)
            log["Metrics/placed_from_finger"] = (self.placed[env_ids] & ~wf).float().sum() / (~wf).float().sum().clamp(min=1)
        log["Metrics/lift_m"], log["Metrics/seconds_to_place"] = f(self.lifted), f(self.t_place)
        log["Metrics/pull1_peak_N"], log["Metrics/pull2_peak_N"] = f(self.peak1), f(self.peak2)
        log["Metrics/seconds_above_110pct_limit"] = f(self.over_time)
        log["Metrics/move_gripper_top_speed_m_s"] = f(self.top_speed)
        log["Metrics/pull_goal_lead_rad"] = (self.qA_tgt - self.A.data.joint_pos.torch)[env_ids][:, self.act_ids].abs().mean()
        log["Metrics/end_sleep_error_rad"] = self._sleep_error()[env_ids].mean()
        log["Metrics/place_tolerance_m"] = torch.tensor(self._place_tol(), device=self.device)
        if len(env_ids) > self.num_envs // 8:  # a plain-text progress line whenever many episodes end together
            import os

            with open(os.path.expanduser("~/team_metrics.log"), "a") as fh:
                fh.write(f"step {self.common_step_counter:6d} " + " ".join(
                    f"{k.split('/')[-1]}={float(v):.3f}" for k, v in log.items() if k.startswith("Metrics/")) + "\n")
        super()._reset_idx(env_ids)
        n, dev, c = len(env_ids), self.device, self.cfg
        origins = self.scene.env_origins[env_ids]
        for arm, default, noise in ((self.A, self.A_default, 0.05), (self.B, self.B_default, 0.03)):
            q = default[env_ids].clone()
            q[:, self.act_ids[1:]] += sample_uniform(-noise, noise, (n, 4), dev)
            q = torch.clamp(q, self.limits[:, 0], self.limits[:, 1])
            root = arm.data.default_root_pose.torch[env_ids].clone()
            root[:, :3] += origins
            arm.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
            arm.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
            arm.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            arm.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
            if arm is self.A:
                self.qA_tgt[env_ids], self.qA_prev[env_ids], self.qA_cmd[env_ids] = q, q, q
            else:
                self.qB_tgt[env_ids], self.qB_prev[env_ids], self.qB_cmd[env_ids] = q, q, q
            k = sample_uniform(c.stiffness_range[0], c.stiffness_range[1], (n, 1), dev).expand(n, len(self.arm_ids))
            d = sample_uniform(c.damping_range[0], c.damping_range[1], (n, 1), dev).expand(n, len(self.arm_ids))
            arm.write_joint_stiffness_to_sim_index(stiffness=k.contiguous(), joint_ids=self.arm_ids, env_ids=env_ids)
            arm.write_joint_damping_to_sim_index(damping=d.contiguous(), joint_ids=self.arm_ids, env_ids=env_ids)
        self.followA[env_ids] = sample_uniform(c.follow_range[0], c.follow_range[1], (n,), dev)
        self.followB[env_ids] = sample_uniform(c.follow_range[0], c.follow_range[1], (n,), dev)
        self.top_speed[env_ids] = 0.0
        rnd = lambda r, *s: sample_uniform(r[0], r[1], (n, *s), dev)
        self.k[env_ids] = c.spring * rnd(c.spring_range)
        self.amp[env_ids], self.ramp[env_ids] = rnd(c.pull_range, 2), rnd(c.ramp_range, 2)
        self.hold[env_ids], self.delay[env_ids] = rnd(c.hold_range, 2), rnd(c.delay_range, 2)
        self.t1[env_ids] = self.delay[env_ids, 0] + 2 * self.ramp[env_ids, 0] + self.hold[env_ids, 0] + 0.3
        self.t2[env_ids] = self.t1[env_ids] + c.move_time
        self.src[env_ids] = torch.randint(0, 2, (n,), device=dev)
        # Two ways to start. Most episodes: the loop is on a finger, which pulls first. The others: the loop lies on
        # the table, there is no first pull, and the move arm must pick it up and put it on the target finger.
        free = torch.rand(n, device=dev) < c.table_start_share
        self.free[env_ids] = free
        if not hasattr(self, "_was_free"):
            self._was_free = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._was_free[env_ids] = free
        self.amp[env_ids, 0] = torch.where(free, torch.zeros(n, device=dev), self.amp[env_ids, 0])
        self.t1[env_ids] = torch.where(free, torch.full((n,), 0.5, device=dev), self.t1[env_ids])
        self.t2[env_ids] = self.t1[env_ids] + c.move_time
        self.fresh[env_ids] = True
        self.seen[env_ids] = True
        for buf in (self.held, self.placed, self.dropped, self.snagged, self.late):
            buf[env_ids] = False
        for buf in (self.grip, self.force, self.force_obs, self.force_prev, self.peak1, self.peak2, self.t_place, self.lifted,
                    self.over_time, self.ev_take, self.ev_place, self.ev_drop, self.ev_snag, self.ev_late):
            buf[env_ids] = 0.0
        for buf in (self.actA, self.last_actA, self.action, self.last_action):
            buf[env_ids] = 0.0
