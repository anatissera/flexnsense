"""Both arms and a physical scale: the move arm learns to put the loop on a finger by contact alone.

Stages of one episode (the move arm is the policy; the pull arm follows a fixed routine):
  0  the loop lies on the table. Move arm: wake up, pinch the tube, put the loop round the INDEX finger, let go.
  1  pull arm: pull back until the scale reads 200 g, hold, return to zero. Move arm: keep clear, rest.
  2  move arm: take the tube again, lift the loop off the index finger, put it round the MIDDLE finger, let go.
  3  pull arm: the same pull. Move arm: back to rest.

No rule moves anything. The loop counts as "on a finger" only when the simulated ring is around the simulated post
(its centre within the ring's free play of the post's axis, and below the fingertip). Rewards only measure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, sample_uniform

from .phys_env_cfg import FINGER_ANGLE, FINGER_TOP, GRASP_AT, LOOP_AT, PULL_BACK, SWIVEL, PhysEnvCfg, finger_xy


class PhysEnv(DirectRLEnv):
    cfg: PhysEnvCfg

    def __init__(self, cfg: PhysEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.A, self.B = self.scene["pull_arm"], self.scene["move_arm"]
        dev, n = self.device, self.num_envs
        ids = lambda arm, expr: torch.tensor(arm.find_joints(expr, preserve_order=True)[0], device=dev)
        self.a_arm = ids(self.A, "rev_motor_0[1-7]")
        self.a_yaw, self.a_pitch, self.a_slide = (int(ids(self.A, k)[0]) for k in ("scale_yaw", "scale_pitch", "scale_slide"))
        self.act_ids, self.arm_ids = ids(self.B, "rev_motor_0[12346]"), ids(self.B, "rev_motor_0[1-7]")
        self.jaw_ids = ids(self.B, "rev_motor_08_[12]")
        self.ee = self.B.find_bodies("Gripper_Base_v1_1")[0][0]
        self.tube_b, self.hook_b = self.A.find_bodies("scale_tube")[0][0], self.A.find_bodies("scale_hook")[0][0]
        self.limits = self.B.data.joint_pos_limits.torch[0]
        self.A_default = self.A.data.default_joint_pos.torch.clone()
        self.B_default = self.B.data.default_joint_pos.torch.clone()
        self.a_home = self.A_default[0, self.a_arm].clone()
        self.a_back = torch.tensor(PULL_BACK, device=dev)
        self.point = torch.tensor([0.0, 1.0, 0.0], device=dev)
        self.swivel = torch.tensor(SWIVEL, device=dev)
        self.fingers = torch.tensor([[*finger_xy(FINGER_ANGLE[k]), 0.02] for k in ("index", "middle")], device=dev)
        z = lambda *s: torch.zeros(n, *s, device=dev)
        self.qA_tgt, self.qB_tgt, self.qB_prev = self.A_default.clone(), self.B_default.clone(), self.B_default.clone()
        self.grip, self.action, self.last_action = z(), z(6), z(6)
        self.stage = torch.zeros(n, dtype=torch.long, device=dev)
        self.started_at = torch.zeros(n, dtype=torch.long, device=dev)
        self.t_stage, self.pull_s, self.pull_hold = z(), z(), z()
        self.pull_mode = torch.zeros(n, dtype=torch.long, device=dev)  # 0 idle, 1 pulling, 2 holding, 3 returning
        self._pull_over = torch.zeros(n, dtype=torch.bool, device=dev)
        self.finished = torch.zeros(n, dtype=torch.bool, device=dev)
        self.late = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ev_place, self.ev_done, self.ev_late = z(), z(), z()
        self.peak, self.tip_prev, self.top_speed, self.lifted, self.far = z(2), z(3), z(), z(), z()

    # ------------------------------------------------------------------ what is where (all measured from the bodies)
    def _where(self):
        o = self.scene.env_origins
        pb = self.B.data.body_pose_w.torch[:, self.ee]
        pointing = matrix_from_quat(pb[:, 3:7]) @ self.point
        tip_b = pb[:, 0:3] + pointing * self.cfg.grip_y - o
        tube = self.A.data.body_pose_w.torch[:, self.tube_b]
        hook = self.A.data.body_pose_w.torch[:, self.hook_b]
        along = matrix_from_quat(tube[:, 3:7]) @ self.point
        grasp = tube[:, 0:3] + along * GRASP_AT - o
        loop = hook[:, 0:3] + (matrix_from_quat(hook[:, 3:7]) @ self.point) * LOOP_AT - o
        force = self.cfg.spring * self.A.data.joint_pos.torch[:, self.a_slide].clamp(min=0.0)
        return tip_b, pointing, grasp, loop, along, force

    def _target(self):
        """The finger the loop belongs on in this stage: index in stages 0 and 1, middle in stages 2 and 3."""
        return self.fingers[(self.stage >= 2).long()]

    def _around(self, loop, finger):
        """The ring is around this post: its centre within the ring's free play of the post, and below the fingertip."""
        return ((loop[:, 0:2] - finger[:, 0:2]).norm(dim=-1) < 0.009) & (loop[:, 2] < FINGER_TOP - 0.008)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        c, dt = self.cfg, self.step_dt
        force = c.spring * self.A.data.joint_pos.torch[:, self.a_slide].clamp(min=0.0)
        # --- the pull arm's routine: back until the scale shows 200 g, hold, return to zero
        m = self.pull_mode
        self.pull_s = torch.where(m == 1, self.pull_s + c.pull_rate * dt, self.pull_s)
        top = (m == 1) & ((force >= c.pull_to) | (self.pull_s >= 1.0))
        self.pull_mode = torch.where(top, torch.full_like(m, 2), self.pull_mode)
        self.pull_hold = torch.where(self.pull_mode == 2, self.pull_hold + dt, torch.zeros_like(self.pull_hold))
        self.pull_mode = torch.where((self.pull_mode == 2) & (self.pull_hold >= c.pull_hold), torch.full_like(m, 3), self.pull_mode)
        self.pull_s = torch.where(self.pull_mode == 3, self.pull_s - c.pull_rate * dt, self.pull_s).clamp(0.0, 1.0)
        self._pull_over = (self.pull_mode == 3) & (self.pull_s <= 0.0)
        self.pull_mode = torch.where(self._pull_over, torch.zeros_like(m), self.pull_mode)
        self.qA_tgt[:, self.a_arm] = self.a_home + self.pull_s.unsqueeze(-1) * (self.a_back - self.a_home)
        # --- the move arm: the policy
        self.last_action = self.action.clone()
        self.action = actions.clamp(-1.0, 1.0)
        q_now = self.B.data.joint_pos.torch[:, self.act_ids]
        q = self.qB_tgt[:, self.act_ids] + c.action_scale * self.action[:, :5]
        q = torch.maximum(torch.minimum(q, q_now + c.max_lead), q_now - c.max_lead)
        self.qB_tgt[:, self.act_ids] = torch.clamp(q, self.limits[self.act_ids, 0], self.limits[self.act_ids, 1])
        self.grip = torch.clamp(self.grip + c.grip_rate * self.action[:, 5], 0.0, 1.0)
        lo, hi = self.limits[self.jaw_ids, 0], self.limits[self.jaw_ids, 1]
        self.qB_tgt[:, self.jaw_ids] = torch.stack(
            [hi[0] + (lo[0] - hi[0]) * self.grip, lo[1] + (hi[1] - lo[1]) * self.grip], dim=-1)

    def _apply_action(self) -> None:
        self.A.set_joint_position_target_index(target=self.qA_tgt)
        self.B.set_joint_position_target_index(target=self.qB_tgt)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        c, dt = self.cfg, self.step_dt
        tip_b, pointing, grasp, loop, along, force = self._where()
        st = self.stage
        working = (st == 0) | (st == 2)
        target = self._target()
        around = self._around(loop, target)
        clear = (self.grip < 0.3) & ((tip_b - grasp).norm(dim=-1) > 0.05)
        placed = working & around & clear
        self.ev_place = placed.float()
        self.pull_mode = torch.where(placed, torch.ones_like(self.pull_mode), self.pull_mode)
        over = ~working & self._pull_over
        self.ev_done = (over & (st == 3) & ~self.finished).float()
        self.finished |= over & (st == 3)
        self.stage = torch.where(placed, st + 1, torch.where(over & (st == 1), torch.full_like(st, 2), st))
        self.t_stage = torch.where(placed | (over & (st == 1)), torch.zeros_like(self.t_stage), self.t_stage + dt)
        late = working & ~placed & (self.t_stage > c.stage_time)
        self.ev_late, self.late = late.float(), late
        pulling = (st == 1) | (st == 3)
        which = (st == 3).long().unsqueeze(-1)
        self.peak.scatter_(1, which, torch.maximum(self.peak.gather(1, which), (force * pulling.float()).unsqueeze(-1)))
        speed = (tip_b - self.tip_prev).norm(dim=-1) / dt
        speed = torch.where(self.episode_length_buf > 1, speed, torch.zeros_like(speed))
        self.top_speed = torch.maximum(self.top_speed, speed)
        self.tip_prev = tip_b.clone()
        self.lifted = torch.maximum(self.lifted, loop[:, 2])
        self.far = torch.maximum(self.far, self.stage.float() + self.finished.float())
        bad = ~torch.isfinite(self.B.data.joint_pos.torch).all(dim=-1) | ~torch.isfinite(self.A.data.joint_pos.torch).all(dim=-1)
        self._now = (tip_b, pointing, grasp, loop, along, force, speed, around, working)
        return late | bad, self.episode_length_buf >= self.max_episode_length

    def _sleep_error(self):
        return (self.B.data.joint_pos.torch - self.B_default)[:, self.arm_ids].abs().mean(dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        c, dt = self.cfg, self.step_dt
        tip_b, pointing, grasp, loop, along, force, speed, around, working = self._now
        near = lambda d, s: 1.0 - torch.tanh(d / s)
        target = self._target()
        w = working.float()
        d_grasp = (tip_b - grasp).norm(dim=-1)
        want = torch.where(around, torch.zeros_like(d_grasp), torch.clamp((0.035 - d_grasp) / 0.02, 0.0, 1.0))
        r_reach = (near(d_grasp, 0.35) + 0.6 * near(d_grasp, 0.08) + 0.5 * near(d_grasp, 0.02)
                   + 0.5 * (1.0 - (self.grip - want).abs()))
        # what is left of the loop's way: up over the fingertip, across to the post, down round it
        safe = FINGER_TOP + 0.014
        d_flat = (loop[:, 0:2] - target[:, 0:2]).norm(dim=-1)
        across = torch.clamp(d_flat - 0.005, min=0.0)
        climb = torch.clamp(safe - loop[:, 2], min=0.0) * torch.clamp(across / 0.005, max=1.0)
        down = torch.where(across > 0, torch.full_like(across, safe - 0.03), torch.clamp(loop[:, 2] - 0.03, min=0.0))
        r_carry = near(climb + across + down, 0.05)
        r_set = around.float() * (1.0 - self.grip)
        asleep = near(self._sleep_error(), 0.25)
        seg = loop - self.swivel
        s = torch.clamp(((tip_b - self.swivel) * seg).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-9), 0.0, 1.0)
        to_scale = (tip_b - (self.swivel + seg * s.unsqueeze(-1))).norm(dim=-1)
        rate = (self.action - self.last_action).square().sum(-1)
        per_second = (
            c.rew_stage * (self.stage.float() + self.finished.float())
            + w * (c.rew_reach * r_reach + c.rew_carry * r_carry + c.rew_set * r_set)
            + (1.0 - w) * (c.rew_sleep * (asleep + 0.2 * (1.0 - self.grip)) + c.rew_intrude * (to_scale < c.keep_away).float())
            + c.rew_rate * rate
            + c.rew_speed * torch.clamp(speed - c.max_tip_speed, min=0.0)
            + c.rew_table * (tip_b[:, 2] < 0.012).float()
            + c.rew_yank * w * torch.clamp(force - 1.0, min=0.0)
        )
        once = c.rew_place_once * self.ev_place + c.rew_all_done_once * self.ev_done + c.rew_late_once * self.ev_late
        return torch.nan_to_num(per_second * dt + once)

    def _get_observations(self) -> dict:
        tip_b, pointing, grasp, loop, along, force = self._where()
        q, c = self.B.data.joint_pos.torch, self.cfg
        qd = (q - self.qB_prev) / self.step_dt
        self.qB_prev = q.clone()
        target = self._target()
        lo, hi = self.limits[self.jaw_ids, 0], self.limits[self.jaw_ids, 1]
        # 1 open, 0 shut: the jaws stop on what they hold, so this tells the policy that something is between them
        gap = ((q[:, self.jaw_ids[0]] - lo[0]) + (hi[1] - q[:, self.jaw_ids[1]])) / 0.051
        obs = torch.cat(
            [
                (q - self.B_default)[:, self.arm_ids], 0.1 * qd[:, self.arm_ids], (self.qB_tgt - q)[:, self.act_ids],
                self.grip.unsqueeze(-1), gap.unsqueeze(-1),
                5.0 * (tip_b - self.swivel), pointing, 5.0 * (grasp - tip_b), 5.0 * (loop - tip_b), 10.0 * (target - loop), along,
                torch.nn.functional.one_hot(self.stage, 4).float(), (self.t_stage / c.stage_time).unsqueeze(-1),
                (force / c.pull_to).unsqueeze(-1), self._around(loop, target).float().unsqueeze(-1),
                self.action,
            ],
            dim=-1,
        )
        return {"policy": torch.nan_to_num(obs).clamp(-20.0, 20.0)}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.A._ALL_INDICES
        log = self.extras.setdefault("log", {})
        f = lambda v: v[env_ids].float().mean()
        full = (self.started_at[env_ids] == 0).float()
        g = lambda v: (v[env_ids].float() * full).sum() / full.sum().clamp(min=1.0)
        log["Metrics/on_index_from_the_table"] = g(self.far >= 1)
        log["Metrics/on_middle_from_the_table"] = g(self.far >= 3)
        log["Metrics/whole_sequence_from_the_table"] = g(self.finished)
        log["Metrics/on_middle_any_start"] = f(self.far >= 3)
        log["Metrics/out_of_time"] = f(self.late)
        log["Metrics/pull1_peak_N"], log["Metrics/pull2_peak_N"] = f(self.peak[:, 0]), f(self.peak[:, 1])
        log["Metrics/loop_highest_m"], log["Metrics/gripper_top_speed_m_s"] = f(self.lifted), f(self.top_speed)
        super()._reset_idx(env_ids)
        n, dev, c = len(env_ids), self.device, self.cfg
        origins = self.scene.env_origins[env_ids]
        on_index = torch.rand(n, device=dev) < c.start_on_index_share
        qa = self.A_default[env_ids].clone()
        spot = FINGER_ANGLE["index"] + sample_uniform(c.table_spot[0], c.table_spot[1], (n,), dev)
        angle = torch.where(on_index, torch.full_like(spot, FINGER_ANGLE["index"]), spot)
        qa[:, self.a_yaw] = -angle  # the swivel's own angle runs the other way round from the bench angle
        qa[:, self.a_pitch] = -math.asin((SWIVEL[2] - 0.004) / LOOP_AT)  # the loop rests on the table
        qa[:, self.a_slide] = 0.0
        qb = self.B_default[env_ids].clone()
        qb[:, self.act_ids[1:]] += sample_uniform(-0.03, 0.03, (n, 4), dev)
        qb = torch.clamp(qb, self.limits[:, 0], self.limits[:, 1])
        for arm, q in ((self.A, qa), (self.B, qb)):
            root = arm.data.default_root_pose.torch[env_ids].clone()
            root[:, :3] += origins
            arm.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
            arm.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
            arm.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            arm.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.qA_tgt[env_ids], self.qB_tgt[env_ids], self.qB_prev[env_ids] = qa, qb, qb
        self.qA_tgt[env_ids, self.a_yaw], self.qA_tgt[env_ids, self.a_pitch] = 0.0, 0.0
        self.stage[env_ids] = torch.where(on_index, 2, 0)
        self.started_at[env_ids] = self.stage[env_ids]
        self.pull_mode[env_ids] = 0
        self.finished[env_ids], self.late[env_ids] = False, False
        for buf in (self.grip, self.t_stage, self.pull_s, self.pull_hold, self.ev_place, self.ev_done, self.ev_late,
                    self.top_speed, self.lifted):
            buf[env_ids] = 0.0
        self.far[env_ids] = self.stage[env_ids].float()
        self.peak[env_ids] = 0.0
        self.action[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
