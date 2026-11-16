"""Both follower arms on one wooden table, each with its own policy, working in sequence.

One episode (20 s, 10 decisions a second for both arms):
  1. A finger pulls on the scale. The PULL arm holds the scale's ring and gives way above the force limit.
     The MOVE arm is asleep (all joints at zero) and keeps clear.
  2. The pull ends. The MOVE arm wakes up, goes to the scale, takes it near its loop, lifts the loop over the finger,
     sets it on the other finger (index <-> middle), lets go and goes back to sleep. The PULL arm keeps holding.
  3. The other finger pulls. The PULL arm limits the force again.

Two policies, one scene:
  * the pull arm runs its own policy, trained separately on the same pull (Flex-PullBench-v0), loaded from a file and
    run here exactly as it will run on the robot (29 numbers in, 5 out);
  * the move arm is the one that learns in this task, next to the working pull arm.

What is simulated and what is a rule: both arms are simulated ElRobots on a table they can collide with. The scale is
the spring model of the pull task while a finger holds its loop. Taking and placing follow the rules of the move task:
jaws closing within 15 mm of the grasp point at the working tilt take the tube, the loop then follows the gripper
(the tube pivots about the ring in the pull arm's jaws), letting go within 8 mm of the target finger places it,
letting go elsewhere drops it, and a loop carried low between the two fingers has hit a finger.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, sample_uniform

from .duo_env_cfg import MOVE_YAW, DuoEnvCfg

PULL_START = [0.0, 0.04, 1.057, 0.072, 0.0, -1.169, 0.0]  # the pull policy's reference posture [rad]


class DuoEnv(DirectRLEnv):
    cfg: DuoEnvCfg

    def __init__(self, cfg: DuoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.A, self.B = self.scene["pull_arm"], self.scene["move_arm"]
        dev, n = self.device, self.num_envs
        ids = lambda expr: torch.tensor(self.A.find_joints(expr, preserve_order=True)[0], device=dev)
        self.act_ids, self.arm_ids, self.jaw_ids = ids("rev_motor_0[12346]"), ids("rev_motor_0[1-7]"), ids("rev_motor_08_[12]")
        self.ee = self.A.find_bodies("Gripper_Base_v1_1")[0][0]
        self.limits = self.A.data.joint_pos_limits.torch[0]
        self.A_default = self.A.data.default_joint_pos.torch.clone()
        self.B_default = self.B.data.default_joint_pos.torch.clone()  # asleep
        self.pull_policy = torch.jit.load(cfg.pull_policy, map_location=dev).eval()
        self.pull_start = torch.tensor(PULL_START, device=dev)
        self.point = torch.tensor([0.0, 1.0, 0.0], device=dev)
        self.up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        # working tilt of the move arm's gripper: 30 degrees forward of straight down, forward being the way it faces
        fwd = (-math.sin(MOVE_YAW), math.cos(MOVE_YAW))
        s30, c30 = math.sin(math.radians(30)), math.cos(math.radians(30))
        self.work_dir = torch.tensor([fwd[0] * s30, fwd[1] * s30, -c30], device=dev)
        z = lambda *s: torch.zeros(n, *s, device=dev)
        b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
        # pull arm
        self.qA_tgt, self.qA_prev = self.A_default.clone(), self.A_default.clone()
        self.actA, self.force, self.force_obs, self.force_prev = z(5), z(), z(), z()
        self.home, self.u0, self.hdir = z(3), z(3), z(3)
        # move arm
        self.qB_tgt, self.qB_prev = self.B_default.clone(), self.B_default.clone()
        self.grip, self.action, self.last_action = z(), z(6), z(6)
        # the bench
        self.stand = z(2, 3)  # where the index (0) and middle (1) finger stand
        self.src = torch.zeros(n, dtype=torch.long, device=dev)
        self.loop, self.take_off = z(3), z(3)
        self.held, self.placed, self.dropped, self.snagged, self.late = b(), b(), b(), b(), b()
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.k, self.amp, self.ramp, self.hold, self.delay = z(), z(2), z(2), z(2), z(2)
        self.t1, self.t2 = z(), z()
        self.ev_take, self.ev_place, self.ev_drop, self.ev_snag, self.ev_late = z(), z(), z(), z(), z()
        self.peak1, self.peak2, self.t_place, self.lifted = z(), z(), z(), z()

    # ------------------------------------------------------------------ geometry and time
    def _tips(self):
        o = self.scene.env_origins
        pa = self.A.data.body_pose_w.torch[:, self.ee]
        pb = self.B.data.body_pose_w.torch[:, self.ee]
        tip_a = pa[:, 0:3] + (matrix_from_quat(pa[:, 3:7]) @ self.point) * self.cfg.tip - o
        pointing_b = matrix_from_quat(pb[:, 3:7]) @ self.point
        tip_b = pb[:, 0:3] + pointing_b * self.cfg.grip_y - o
        return tip_a, tip_b, pointing_b

    def _time(self):
        return self.episode_length_buf * self.step_dt

    def _phase(self):
        """0: first pull, 1: the move, 2: second pull."""
        t = self._time()
        return (t >= self.t1).long() + (t >= self.t2).long()

    def _pull_now(self):
        """How far the pulling finger has moved away [m], and which finger pulls (source in the first pull)."""
        t, ph = self._time(), self._phase()
        which = (ph == 2).long()  # 0: first pull's numbers, 1: second pull's
        pick = lambda v: v.gather(1, which.unsqueeze(-1)).squeeze(-1)
        tt = torch.where(ph == 2, t - self.t2, t) - pick(self.delay)
        ramp, hold = pick(self.ramp), pick(self.hold)
        sm = lambda p: (lambda c: c * c * (3 - 2 * c))(torch.clamp(p, 0.0, 1.0))
        pull = pick(self.amp) * (sm(tt / ramp) - sm((tt - ramp - hold) / ramp))
        return torch.where(ph == 1, torch.zeros_like(pull), pull)

    def _finger(self, idx: torch.Tensor):
        """Where finger idx stands, and the level direction from the ring's home to it."""
        stand = self.stand.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        u = stand - self.home
        u[:, 2] = 0.0
        return stand, u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _on_finger(self):
        """The finger the loop is on when it is not carried: the source until placed, then the target."""
        return torch.where(self.placed, 1 - self.src, self.src)

    def _place_tol(self) -> float:
        c = self.cfg
        done = min(1.0, self.common_step_counter / max(1, c.place_anneal_steps))
        return c.place_within_start + (c.place_within - c.place_within_start) * done

    def _grasp_point(self, tip_a: torch.Tensor):
        d = self.loop - tip_a
        return tip_a + d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.cfg.grasp_at

    def _set_home(self, tip_a: torch.Tensor):
        ids = self.fresh.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        c = self.cfg
        self.home[ids] = tip_a[ids]
        u = tip_a[ids].clone()
        u[:, 2] = 0.0
        u = u / u.norm(dim=-1, keepdim=True)
        self.u0[ids] = u
        self.hdir[ids] = torch.linalg.cross(self.up.expand(len(ids), 3), u)
        for j in range(2):  # index straight ahead of the ring, middle finger 24 mm beside it
            self.stand[ids, j] = tip_a[ids] + u * c.rest + self.hdir[ids] * c.finger_gap * j
        self.loop[ids] = self.stand[ids, self.src[ids]]
        free = getattr(self, "free", None)
        if free is not None and bool(free[ids].any()):
            # the other start: the loop lies on the table beside the index finger, not on any finger. The tube keeps
            # its length, so the spot is on a circle of that radius round the ring.
            m = len(ids)
            flat = (tip_a[ids] + u * (c.rest - sample_uniform(0.0, 0.03, (m, 1), self.device))
                    - self.hdir[ids] * sample_uniform(0.045, 0.07, (m, 1), self.device))
            flat[:, 2] = 0.012
            v = flat - tip_a[ids]
            spot = tip_a[ids] + v / v.norm(dim=-1, keepdim=True) * c.rest
            self.loop[ids] = torch.where(free[ids].unsqueeze(-1), spot, self.loop[ids])
        self.fresh[ids] = False

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        c = self.cfg
        tip_a, _, _ = self._tips()
        live = ~self.fresh
        on = self._on_finger()
        stand, u = self._finger(on)
        # --- the pull arm: its own policy, fed exactly what it was trained on
        qa = self.A.data.joint_pos.torch
        side = torch.linalg.cross(self.up.expand_as(u), u)
        off = torch.where(live.unsqueeze(-1), tip_a - self.home, torch.zeros_like(tip_a))
        reading = self.force + c.force_noise * torch.randn_like(self.force)
        reading = torch.round(reading / c.force_step) * c.force_step
        reading = torch.where(torch.rand_like(reading) < c.force_dropout, self.force_obs, reading)
        self.force_prev, self.force_obs = self.force_obs.clone(), reading
        obs_a = torch.cat(
            [
                qa[:, self.arm_ids] - self.pull_start,
                0.1 * (qa - self.qA_prev)[:, self.arm_ids] / self.step_dt,
                (self.qA_tgt - qa)[:, self.act_ids],
                (self.force_prev / c.cap).unsqueeze(-1),
                ((self.force_obs - self.force_prev) / c.cap).unsqueeze(-1),
                10.0 * torch.stack([(off * u).sum(-1), (off * side).sum(-1), off[:, 2]], dim=-1),
                self.actA,
            ],
            dim=-1,
        )
        self.qA_prev = qa.clone()
        with torch.no_grad():
            act_a = self.pull_policy(obs_a).clamp(-1.0, 1.0)
        self.actA = torch.where(live.unsqueeze(-1), act_a, torch.zeros_like(act_a))
        q = self.qA_tgt[:, self.act_ids] + c.pull_action_scale * self.actA
        self.qA_tgt[:, self.act_ids] = torch.clamp(q, self.limits[self.act_ids, 0], self.limits[self.act_ids, 1])
        # --- the scale: a spring from the pull arm's jaws to the finger its loop is on; no pull while it is carried
        finger = stand + u * self._pull_now().unsqueeze(-1)
        self.loop = torch.where((live & ~self.held).unsqueeze(-1), finger, self.loop)
        d = self.loop - tip_a
        dist = d.norm(dim=-1).clamp(min=1e-6)
        force = self.k * torch.clamp(dist - c.rest, min=0.0)
        self.force = torch.where(live & ~self.held, force, torch.zeros_like(force))
        self._wrench = (d / dist.unsqueeze(-1) * self.force.unsqueeze(-1)).unsqueeze(1)
        self._wrench_at = (tip_a + self.scene.env_origins).unsqueeze(1)
        # --- the move arm: the policy that learns here
        self.last_action = self.action.clone()
        self.action = actions.clamp(-1.0, 1.0)
        q = self.qB_tgt[:, self.act_ids] + c.action_scale * self.action[:, :5]
        self.qB_tgt[:, self.act_ids] = torch.clamp(q, self.limits[self.act_ids, 0], self.limits[self.act_ids, 1])
        self.grip = torch.clamp(self.grip + c.grip_rate * self.action[:, 5], 0.0, 1.0)
        lo, hi = self.limits[self.jaw_ids, 0], self.limits[self.jaw_ids, 1]
        shut = torch.where(self.held, self.grip.clamp(max=0.62), self.grip)  # a held tube stops the jaws at about 60 %
        self.qB_tgt[:, self.jaw_ids] = torch.stack([hi[0] + (lo[0] - hi[0]) * shut, lo[1] + (hi[1] - lo[1]) * shut], dim=-1)

    def _apply_action(self) -> None:
        self.A.set_joint_position_target_index(target=self.qA_tgt)
        self.B.set_joint_position_target_index(target=self.qB_tgt)
        self.A.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._wrench, positions=self._wrench_at, body_ids=[self.ee], is_global=True
        )

    def _rules(self):
        """Take, carry, place, drop, hit a finger, too late. Once per step, after the physics."""
        c = self.cfg
        tip_a, tip_b, pointing = self._tips()
        self._set_home(tip_a)
        ph = self._phase()
        live = ~self.placed & ~self.dropped & ~self.snagged & ~self.late
        tilt = torch.acos(torch.clamp((pointing * self.work_dir).sum(-1), -1.0, 1.0))
        grasp = self._grasp_point(tip_a)
        near = (tip_b - grasp).norm(dim=-1) < c.take_within
        take = live & ~self.held & (ph == 1) & near & (self.grip > 0.6) & (tilt < c.take_tilt)
        self.held |= take
        self.take_off = torch.where(take.unsqueeze(-1), grasp - tip_b, self.take_off)
        prev_x = (self.loop * self.hdir).sum(-1)  # position across the fingers
        prev_low = self.loop[:, 2] < self.home[:, 2] + c.clear_height
        d = tip_b + self.take_off - tip_a
        carried = tip_a + d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6) * c.rest
        self.loop = torch.where(self.held.unsqueeze(-1), carried, self.loop)
        dst, _ = self._finger(1 - self.src)
        release = live & self.held & (self.grip < 0.35) & ~take
        at_goal = (self.loop - dst).norm(dim=-1) < self._place_tol()
        place, drop = release & at_goal, release & ~at_goal
        # a loop that is low while between the two fingers has hit one (checked over the whole step)
        x = (self.loop * self.hdir).sum(-1)
        xs, xd = (self.stand[:, 0] * self.hdir).sum(-1), (self.stand[:, 1] * self.hdir).sum(-1)
        band_lo, band_hi = torch.minimum(xs, xd) + 0.006, torch.maximum(xs, xd) - 0.006
        low = (self.loop[:, 2] < self.home[:, 2] + c.clear_height) | prev_low
        crossing = (torch.maximum(prev_x, x) > band_lo) & (torch.minimum(prev_x, x) < band_hi)
        snag = live & self.held & low & crossing & ~release
        late = live & (ph == 2)  # the second pull starts and the loop is not on its finger
        self.held &= ~(release | snag | late)
        self.placed |= place
        if hasattr(self, "free"):
            self.free &= ~place  # once placed it is on a finger like any other
        self.dropped |= drop
        self.snagged |= snag
        self.late |= late
        self.ev_take, self.ev_place, self.ev_drop = take.float(), place.float(), drop.float()
        self.ev_snag, self.ev_late = snag.float(), late.float()
        self.t_place = torch.where(place, self._time() - self.t1, self.t_place)
        self.lifted = torch.maximum(self.lifted, (self.loop[:, 2] - self.home[:, 2]) * self.held.float())
        self.peak1 = torch.maximum(self.peak1, self.force * (ph == 0).float())
        self.peak2 = torch.maximum(self.peak2, self.force * (ph == 2).float())
        self._now = (tip_a, tip_b, pointing, ph, dst)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._rules()
        return self.dropped | self.snagged | self.late, self.episode_length_buf >= self.max_episode_length

    def _sleep_error(self):
        q = self.B.data.joint_pos.torch
        return (q - self.B_default)[:, self.arm_ids].abs().mean(dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        c, dt = self.cfg, self.step_dt
        tip_a, tip_b, pointing, ph, dst = self._now
        near_fn = lambda d, s: 1.0 - torch.tanh(d / s)
        asleep = near_fn(self._sleep_error(), 0.25)
        moving = (ph == 1) & ~self.placed
        searching = (moving & ~self.held).float()
        waiting = ((ph != 1) & ~self.placed).float()  # the first pull: stay asleep
        d_grasp = (tip_b - self._grasp_point(tip_a)).norm(dim=-1)
        want = torch.clamp((0.035 - d_grasp) / 0.02, 0.0, 1.0)
        # three ranges, so that there is always something to follow: from the resting posture the scale is about
        # 40 cm away, where a short-range term alone is flat and the arm never finds its way
        r_reach = (near_fn(d_grasp, 0.35) + 0.6 * near_fn(d_grasp, 0.08) + 0.5 * near_fn(d_grasp, 0.02)
                   + 0.5 * (1.0 - (self.grip - want).abs()))
        aim = 1.0 - (pointing * self.work_dir).sum(-1)
        d_dst = (self.loop - dst).norm(dim=-1)
        d_flat = (self.loop[:, 0:2] - dst[:, 0:2]).norm(dim=-1)
        safe = c.clear_height + 0.012
        z = self.loop[:, 2] - dst[:, 2]
        across = torch.clamp(d_flat - 0.003, min=0.0)
        climb = torch.clamp(safe - z, min=0.0) * torch.clamp(across / 0.004, max=1.0)
        remaining = climb + across + torch.where(across > 0, torch.full_like(z, safe), z.abs())
        r_carry = 1.0 + 2.5 * near_fn(remaining, 0.05) + 4.0 * near_fn(d_dst, 0.012) * (1.0 - self.grip)
        r_done = 1.0 + asleep + 0.2 * (1.0 - self.grip)
        # during a pull the move arm keeps its fingers away from the scale
        seg = self.loop - tip_a
        s = torch.clamp(((tip_b - tip_a) * seg).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-9), 0.0, 1.0)
        to_scale = (tip_b - (tip_a + seg * s.unsqueeze(-1))).norm(dim=-1)
        intrude = ((ph != 1) & (to_scale < c.keep_away)).float()
        rate = (self.action - self.last_action).square().sum(-1)
        per_second = (
            c.rew_sleep * waiting * (asleep + 0.2 * (1.0 - self.grip))
            + c.rew_reach * searching * r_reach
            + c.rew_carry * self.held.float() * r_carry
            + c.rew_done * self.placed.float() * r_done
            + c.rew_aim * (searching + self.held.float()) * aim
            + c.rew_intrude * intrude
            + c.rew_rate * rate
            + c.rew_table * (tip_b[:, 2] < 0.015).float()
        )
        once = (c.rew_take_once * self.ev_take + c.rew_place_once * self.ev_place + c.rew_drop_once * self.ev_drop
                + c.rew_snag_once * self.ev_snag + c.rew_late_once * self.ev_late)
        self._d_grasp, self._d_dst = d_grasp, d_dst
        return per_second * dt + once

    def _get_observations(self) -> dict:
        tip_a, tip_b, pointing = self._tips()
        ph = self._phase()
        q, c = self.B.data.joint_pos.torch, self.cfg
        qd = (q - self.qB_prev) / self.step_dt
        self.qB_prev = q.clone()
        dst, _ = self._finger(1 - self.src)
        src, _ = self._finger(self.src)
        t_in = torch.where(ph == 1, self._time() - self.t1, torch.zeros_like(self.t1)) / c.move_time
        obs = torch.cat(
            [
                (q - self.B_default)[:, self.arm_ids],
                0.1 * qd[:, self.arm_ids],
                (self.qB_tgt - q)[:, self.act_ids],
                self.grip.unsqueeze(-1),
                5.0 * (tip_b - tip_a),
                pointing,
                5.0 * (self._grasp_point(tip_a) - tip_b),
                5.0 * (self.loop - tip_b),
                10.0 * (dst - self.loop),
                10.0 * (dst - src),
                self.held.float().unsqueeze(-1),
                self.placed.float().unsqueeze(-1),
                torch.nn.functional.one_hot(ph, 3).float(),
                t_in.unsqueeze(-1),
                (self.force_obs / c.cap).unsqueeze(-1),
                self.action,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.A._ALL_INDICES
        log = self.extras.setdefault("log", {})
        f = lambda v: v[env_ids].float().mean()
        log["Metrics/placed"], log["Metrics/dropped"] = f(self.placed), f(self.dropped)
        log["Metrics/hit_a_finger"], log["Metrics/too_late"] = f(self.snagged), f(self.late)
        log["Metrics/taken"] = f(self.held | self.placed | self.dropped | self.snagged)
        log["Metrics/lift_m"], log["Metrics/seconds_to_place"] = f(self.lifted), f(self.t_place)
        log["Metrics/pull1_peak_N"], log["Metrics/pull2_peak_N"] = f(self.peak1), f(self.peak2)
        log["Metrics/end_sleep_error_rad"] = self._sleep_error()[env_ids].mean()
        log["Metrics/place_tolerance_m"] = torch.tensor(self._place_tol(), device=self.device)
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
                self.qA_tgt[env_ids], self.qA_prev[env_ids] = q, q
            else:
                self.qB_tgt[env_ids], self.qB_prev[env_ids] = q, q
        rnd = lambda r, *s: sample_uniform(r[0], r[1], (n, *s), dev)
        self.k[env_ids] = c.spring * rnd(c.spring_range)
        self.amp[env_ids], self.ramp[env_ids] = rnd(c.pull_range, 2), rnd(c.ramp_range, 2)
        self.hold[env_ids], self.delay[env_ids] = rnd(c.hold_range, 2), rnd(c.delay_range, 2)
        self.t1[env_ids] = self.delay[env_ids, 0] + 2 * self.ramp[env_ids, 0] + self.hold[env_ids, 0] + 0.3
        self.t2[env_ids] = self.t1[env_ids] + c.move_time
        self.src[env_ids] = torch.randint(0, 2, (n,), device=dev)
        self.fresh[env_ids] = True
        for buf in (self.held, self.placed, self.dropped, self.snagged, self.late):
            buf[env_ids] = False
        for buf in (self.grip, self.force, self.force_obs, self.force_prev, self.peak1, self.peak2, self.t_place, self.lifted,
                    self.ev_take, self.ev_place, self.ev_drop, self.ev_snag, self.ev_late):
            buf[env_ids] = 0.0
        self.actA[env_ids] = 0.0
        self.action[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
