"""The move arm as a training task: take the scale near its loop and move the loop from one finger to another.

What is simulated and what is a rule:
  * The arm is the simulated ElRobot (same model and servo settings as the pull task).
  * The scale is NOT a body here. Its ring end stays where the pull arm holds it. Its loop end is either on a finger,
    or carried: when the jaws close within 15 mm of the grasp point, at the working tilt, the tube is "taken" and its
    loop then follows the gripper (the tube pivots about the ring). Letting go within 8 mm of the target finger places
    the loop; letting go anywhere else drops it and ends the episode. A loop carried lower than 2 cm above its resting
    height while between two fingers has hit a finger: that also ends the episode as a failure.
    The physical pinch itself is shown to work in sim/08_physical_scale.py; here it is a rule so that 4096 arms can
    learn the movement quickly.

What the policy sees (44 numbers): the arm joints (position, speed), distance of the driven joints from their targets,
the gripper command, where the point between the fingers is and where it points, the grasp point, the loop and the
target finger relative to it, whether the tube is held or placed, and its last action.
What it does (6 numbers): changes to the targets of motors 1, 2, 3, 4, 6, and open/close.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, sample_uniform

from .move_env_cfg import MoveEnvCfg


class MoveEnv(DirectRLEnv):
    cfg: MoveEnvCfg

    def __init__(self, cfg: MoveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        dev, n = self.device, self.num_envs
        ids = lambda expr: torch.tensor(self.robot.find_joints(expr, preserve_order=True)[0], device=dev)
        self.act_ids, self.arm_ids = ids("rev_motor_0[12346]"), ids("rev_motor_0[1-7]")
        self.jaw_ids = ids("rev_motor_08_[12]")
        self.ee = self.robot.find_bodies("Gripper_Base_v1_1")[0][0]
        self.limits = self.robot.data.joint_pos_limits.torch[0]
        self.default_q = self.robot.data.default_joint_pos.torch.clone()
        self.q_tgt = self.default_q.clone()
        self.point = torch.tensor([0.0, 1.0, 0.0], device=dev)
        # working tilt: 30 degrees forward of straight down; the arm faces -x in its environment
        self.work_dir = torch.tensor([-math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30))], device=dev)
        self.up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        z = lambda *s: torch.zeros(n, *s, device=dev)
        self.ring, self.src, self.dst, self.loop, self.park = z(3), z(3), z(3), z(3), z(3)
        self.take_off = z(3)  # grasp point minus gripper point at the moment of taking
        self.finger0 = z(3)  # where the first finger of the row stands (for drawing)
        self.grip = z()  # 0 open ... 1 closed (the command)
        self.held = torch.zeros(n, dtype=torch.bool, device=dev)
        self.placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.snagged = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.action, self.last_action = z(6), z(6)
        self.ev_take, self.ev_place, self.ev_drop, self.drag = z(), z(), z(), z()
        self.t_place, self.lifted = z(), z()

    # ------------------------------------------------------------------ helpers
    def _tip(self):
        pose = self.robot.data.body_pose_w.torch[:, self.ee]
        pointing = matrix_from_quat(pose[:, 3:7]) @ self.point
        return pose[:, 0:3] + pointing * self.cfg.grip_y - self.scene.env_origins, pointing  # in the environment's frame

    def _place_tol(self) -> float:
        """How close to the target finger a release counts as placed: generous at first, then the real tolerance."""
        cfg = self.cfg
        done = min(1.0, self.common_step_counter / max(1, cfg.place_anneal_steps))
        return cfg.place_within_start + (cfg.place_within - cfg.place_within_start) * done

    def _grasp_point(self):
        d = self.loop - self.ring
        return self.ring + d / d.norm(dim=-1, keepdim=True) * self.cfg.grasp_at

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        self.last_action = self.action.clone()
        self.action = actions.clamp(-1.0, 1.0)
        q = self.q_tgt[:, self.act_ids] + cfg.action_scale * self.action[:, :5]
        self.q_tgt[:, self.act_ids] = torch.clamp(q, self.limits[self.act_ids, 0], self.limits[self.act_ids, 1])
        self.grip = torch.clamp(self.grip + cfg.grip_rate * self.action[:, 5], 0.0, 1.0)
        lo, hi = self.limits[self.jaw_ids, 0], self.limits[self.jaw_ids, 1]
        # jaw 08_1 runs 0 -> -25.5 mm to close, jaw 08_2 runs 0 -> +25.5 mm; a held tube stops them at about 60 %
        shut = torch.where(self.held, self.grip.clamp(max=0.62), self.grip).unsqueeze(-1)
        self.q_tgt[:, self.jaw_ids] = torch.stack([hi[0] + (lo[0] - hi[0]) * shut[:, 0], lo[1] + (hi[1] - lo[1]) * shut[:, 0]], dim=-1)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(target=self.q_tgt)

    def _update_scale(self):
        """Apply the rules: take, carry, place, drop. Called once per step, after the physics."""
        cfg = self.cfg
        tip, pointing = self._tip()
        live = ~self.fresh & ~self.placed & ~self.dropped & ~self.snagged
        closing, opening = self.grip > 0.6, self.grip < 0.35
        tilt = torch.acos(torch.clamp((pointing * self.work_dir).sum(-1), -1.0, 1.0))
        near = (tip - self._grasp_point()).norm(dim=-1) < cfg.take_within
        take = live & ~self.held & near & closing & (tilt < cfg.take_tilt)
        self.held |= take
        self.ev_take = take.float()
        # The tube keeps the offset it was taken with: without this the loop jumped onto the gripper's line at the
        # moment of taking, and the policy learned to grab off-centre so that the loop landed on the next finger.
        self.take_off = torch.where(take.unsqueeze(-1), self._grasp_point() - tip, self.take_off)
        prev_x = self.loop[:, 0].clone()
        prev_low = self.loop[:, 2] < self.ring[:, 2] + cfg.clear_height
        # carried: the tube pivots about the ring and its loop follows the gripper
        d = tip + self.take_off - self.ring
        carried = self.ring + d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6) * cfg.scale_len
        self.loop = torch.where(self.held.unsqueeze(-1), carried, self.loop)
        release = live & self.held & opening & ~take
        at_goal = (self.loop - self.dst).norm(dim=-1) < self._place_tol()
        place, drop = release & at_goal, release & ~at_goal
        self.held &= ~release
        self.placed |= place
        self.dropped |= drop
        self.loop = torch.where(place.unsqueeze(-1), self.dst, self.loop)
        self.ev_place, self.ev_drop = place.float(), drop.float()
        self.t_place = torch.where(place, self.episode_length_buf * self.step_dt, self.t_place)
        # the loop pushed through a finger: low, and not over the finger it left or the one it goes to
        # (checked over the whole step, so that a fast flick cannot jump across a finger between two looks)
        low = (self.loop[:, 2] < self.ring[:, 2] + cfg.clear_height) | prev_low
        band_lo = torch.minimum(self.src[:, 0], self.dst[:, 0]) + 0.006
        band_hi = torch.maximum(self.src[:, 0], self.dst[:, 0]) - 0.006
        x_lo, x_hi = torch.minimum(prev_x, self.loop[:, 0]), torch.maximum(prev_x, self.loop[:, 0])
        off = (x_hi > band_lo) & (x_lo < band_hi)
        # A finger stands in the way: a loop that is low and between fingers has hit one. That is a failure and ends
        # the episode (a small penalty was not enough: the policy learned to swipe the loop straight through).
        snag = self.held & low & off
        self.drag = snag.float()
        self.snagged |= snag
        self.held &= ~snag
        self.lifted = torch.maximum(self.lifted, (self.loop[:, 2] - self.ring[:, 2]) * self.held.float())
        self.fresh[:] = False
        return tip, pointing

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._tip_now, self._pointing_now = self._update_scale()
        return self.dropped | self.snagged, self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self) -> torch.Tensor:
        cfg, dt = self.cfg, self.step_dt
        tip, pointing = self._tip_now, self._pointing_now
        near_fn = lambda d, s: 1.0 - torch.tanh(d / s)
        searching = (~self.held & ~self.placed).float()
        d_grasp = (tip - self._grasp_point()).norm(dim=-1)
        # reach with open jaws; close once there
        # the jaws should be open far away and close gradually over the last 3 cm
        want = torch.clamp((0.035 - d_grasp) / 0.02, 0.0, 1.0)
        r_reach = near_fn(d_grasp, 0.06) + 0.5 * near_fn(d_grasp, 0.015) + 0.5 * (1.0 - (self.grip - want).abs())
        self._d_grasp = d_grasp
        # carry: the reward follows the route the loop must take: up to a safe height, across, down onto the target.
        # "remaining" is the length of that route still to go, so every step along it pays more and there is no
        # place off the route that pays better (two earlier versions had such places, and the policy parked there).
        d_dst = (self.loop - self.dst).norm(dim=-1)
        d_flat = (self.loop[:, 0:2] - self.dst[:, 0:2]).norm(dim=-1)
        safe = cfg.clear_height + 0.012
        z = self.loop[:, 2] - self.dst[:, 2]
        across = torch.clamp(d_flat - 0.003, min=0.0)
        climb = torch.clamp(safe - z, min=0.0) * torch.clamp(across / 0.004, max=1.0)  # only matters while still to the side
        remaining = climb + across + torch.where(across > 0, torch.full_like(z, safe), z.abs())
        # over the target, opening the jaws must clearly pay: with a weak term here the policy carried the loop to
        # 9 mm from the target and then held it there, jaws shut, for the rest of the episode
        r_carry = 1.0 + 2.5 * near_fn(remaining, 0.05) + 4.0 * near_fn(d_dst, 0.012) * (1.0 - self.grip)
        self._d_dst = d_dst
        # done: let go, go back to the parking place
        r_done = 1.0 + 0.3 * near_fn((tip - self.park).norm(dim=-1), 0.08) + 0.2 * (1.0 - self.grip)
        aim = 1.0 - (pointing * self.work_dir).sum(-1)
        rate = (self.action - self.last_action).square().sum(-1)
        per_second = (
            cfg.rew_reach * searching * r_reach
            + cfg.rew_carry * self.held.float() * r_carry
            + cfg.rew_done * self.placed.float() * r_done
            + cfg.rew_drag * self.drag
            + cfg.rew_aim * aim
            + cfg.rew_rate * rate
            + cfg.rew_table * (tip[:, 2] < 0.015).float()
        )
        once = cfg.rew_take_once * self.ev_take + cfg.rew_place_once * self.ev_place + cfg.rew_drop_once * self.ev_drop
        return per_second * dt + once

    def _get_observations(self) -> dict:
        tip, pointing = self._tip()
        q, qd = self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch
        obs = torch.cat(
            [
                (q - self.default_q)[:, self.arm_ids],
                0.1 * qd[:, self.arm_ids],
                (self.q_tgt - q)[:, self.act_ids],
                self.grip.unsqueeze(-1),
                5.0 * (tip - self.ring),
                pointing,
                10.0 * (self._grasp_point() - tip),
                10.0 * (self.loop - tip),
                10.0 * (self.dst - self.loop),
                10.0 * (self.dst - self.src),
                self.held.float().unsqueeze(-1),
                self.placed.float().unsqueeze(-1),
                self.action,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        log = self.extras.setdefault("log", {})
        log["Metrics/placed"] = self.placed[env_ids].float().mean()
        log["Metrics/dropped"] = self.dropped[env_ids].float().mean()
        log["Metrics/hit_a_finger"] = self.snagged[env_ids].float().mean()
        log["Metrics/taken_or_placed"] = (self.held[env_ids] | self.placed[env_ids] | self.dropped[env_ids]).float().mean()
        log["Metrics/lift_m"] = self.lifted[env_ids].mean()
        if hasattr(self, "_d_grasp"):
            log["Metrics/end_distance_to_grasp_m"] = self._d_grasp[env_ids].mean()
            log["Metrics/end_tilt_error_rad"] = torch.acos(torch.clamp((self._pointing_now[env_ids] * self.work_dir).sum(-1), -1, 1)).mean()
            log["Metrics/end_grip"] = self.grip[env_ids].mean()
            log["Metrics/end_loop_to_target_m"] = self._d_dst[env_ids].mean()
            log["Metrics/place_tolerance_m"] = torch.tensor(self._place_tol(), device=self.device)
        super()._reset_idx(env_ids)
        n, dev, cfg = len(env_ids), self.device, self.cfg
        q = self.default_q[env_ids].clone()
        q[:, self.act_ids] += sample_uniform(-0.08, 0.08, (n, 5), dev)
        q = torch.clamp(q, self.limits[:, 0], self.limits[:, 1])
        root = self.robot.data.default_root_pose.torch[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.q_tgt[env_ids] = q
        # the bench: ring, the finger the loop is on, the finger it must go to (a neighbour)
        ring = torch.tensor(cfg.ring, device=dev).expand(n, 3) + sample_uniform(-cfg.bench_noise, cfg.bench_noise, (n, 3), dev) * torch.tensor([1.0, 1.0, 0.3], device=dev)
        src_i = torch.randint(0, 4, (n,), device=dev)
        step = torch.where(torch.rand(n, device=dev) < 0.5, -1, 1)
        dst_i = src_i + step
        dst_i = torch.where((dst_i < 0) | (dst_i > 3), src_i - step, dst_i)
        hand_x = sample_uniform(-cfg.bench_noise, cfg.bench_noise, (n,), dev)

        def finger(i):
            x = ring[:, 0] + hand_x + cfg.finger_gap * i.float()
            reach = torch.sqrt(torch.clamp(cfg.scale_len**2 - (x - ring[:, 0]) ** 2, min=1e-6))
            return torch.stack([x, ring[:, 1] + reach, ring[:, 2]], dim=-1)

        self.ring[env_ids], self.src[env_ids], self.dst[env_ids] = ring, finger(src_i), finger(dst_i)
        self.loop[env_ids] = self.src[env_ids]
        self.finger0[env_ids] = finger(torch.zeros_like(src_i))
        self.park[env_ids] = self.ring[env_ids] + torch.tensor([0.02, cfg.grasp_at, 0.06], device=dev)
        for buf in (self.held, self.placed, self.dropped, self.snagged):
            buf[env_ids] = False
        self.fresh[env_ids] = True
        for buf in (self.grip, self.ev_take, self.ev_place, self.ev_drop, self.drag, self.t_place, self.lifted):
            buf[env_ids] = 0.0
        self.action[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
