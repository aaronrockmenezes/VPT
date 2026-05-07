# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""A* data collector — navigates envs, saves images only on success.

Output structure (mirrors data-collection pipeline + `rollout/` subfolder):
  {base_path}/RGB/{Yes|No}/rollout/env_{folder_idx}/step_{N:05d}_{a}.png
  {base_path}/Semantic/{Yes|No}/rollout/env_{folder_idx}/step_{N:05d}_{a}.png
  {base_path}/RGB/{Yes|No}/rollout/env_{folder_idx}/final_pos_*.png
  {base_path}/Semantic/{Yes|No}/rollout/env_{folder_idx}/final_pos_*.png
  {base_path}/cam/{Yes|No}/rollout/env_{folder_idx}/final_cam_semantic.png
  {base_path}/cam/{Yes|No}/rollout/env_{folder_idx}/actions.txt
  {base_path}/cam/{Yes|No}/rollout/env_{folder_idx}/meta.json

Also maintains `{base_path}/successful_envs.json` flushed on every success.

Per RL-reset: agent respawns in 10x10 zone around camera, 4x4 deadzone.
50/25/25 quota enforced via env label pool; overflow successes discarded.
"""

import argparse
import json
import math
import os
import random
import shutil
import signal
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher


def force_exit(signum, frame):
    print(f"\nForce killing self (PID: {os.getpid()})...")
    os.kill(os.getpid(), signal.SIGKILL)


signal.signal(signal.SIGINT, force_exit)
signal.signal(signal.SIGTSTP, force_exit)

parser = argparse.ArgumentParser(description="A* data collector.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="VPT-v18-A-star")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--target_successes", type=int, default=10000)
parser.add_argument("--pos_tol", type=float, default=0.2)
parser.add_argument("--yaw_tol_deg", type=float, default=11.46)
parser.add_argument("--max_plan_steps", type=int, default=512)
parser.add_argument("--max_plan_attempts", type=int, default=4)
parser.add_argument("--max_total_steps", type=int, default=150)
parser.add_argument("--max_align_steps", type=int, default=50)
parser.add_argument("--spawn_half_extent", type=float, default=5.0)
parser.add_argument("--spawn_deadzone", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import cube_game.tasks  # noqa: F401

import gymnasium as gym
import torch

NOOP_ACTION = 4
ACTION_LABELS = {0: "fwd", 1: "back", 2: "left", 3: "right", 4: "noop"}

NAVIGATING   = 0
ALIGNING_YAW = 1
DONE_STATE   = 2

TARGET_FRACTIONS = {"in_view": 0.50, "occluded": 0.25, "outside_fov": 0.25}


# ── helpers ──────────────────────────────────────────────────────────────────

def _compute_yaw_err(u, env_idx):
    return abs(math.degrees(u._normalize_angle(
        u._get_camera_corrected_yaw(env_idx) - u._get_agent_yaw(env_idx))))


def _reason_for(u, i):
    return u.env_visibility_reasons.get(u.slot_folder_indices[i], "unknown")


def _label_for(u, i):
    return u.env_visibility_labels.get(u.slot_folder_indices[i], "unknown")


def _ep_dirs(u, folder_idx, label):
    """Return RGB / Semantic / cam rollout dirs for given folder_idx."""
    base = f"{u.base_path}/{{}}/{label}/rollout/env_{folder_idx}"
    return {"rgb": base.format("RGB"),
            "semantic": base.format("Semantic"),
            "cam": base.format("cam")}


def _spawn_agent_near_camera(u, env_idx, rng, half_ext, deadzone):
    """Teleport agent to collision-free XY in [camera ± half_ext] \ deadzone."""
    cam_xy = u._camera_obj.data.root_pos_w[env_idx, :2].cpu()
    cx, cy = float(cam_xy[0]), float(cam_xy[1])
    env_origin = u.scene.env_origins[env_idx, :2].cpu()
    ox, oy = float(env_origin[0]), float(env_origin[1])
    z = float(u._agent.data.default_root_state[env_idx, 2])
    ids_t = torch.tensor([env_idx], device=u.device, dtype=torch.long)

    for _ in range(200):
        dx = rng.uniform(-half_ext, half_ext)
        dy = rng.uniform(-half_ext, half_ext)
        if abs(dx) < deadzone and abs(dy) < deadzone:
            continue
        wx = max(ox - 9.5, min(ox + 9.5, cx + dx))
        wy = max(oy - 9.5, min(oy + 9.5, cy + dy))
        yaw = rng.uniform(-math.pi, math.pi)
        quat = u._yaw_to_quat_wxyz(yaw).unsqueeze(0)
        pos_t = torch.tensor([[wx, wy, z]], device=u.device, dtype=torch.float32)
        if not u._check_collisions_vectorized(ids_t, pos_t, quat)[0]:
            pose_t = torch.cat([pos_t, quat], dim=1)
            u._agent.write_root_com_pose_to_sim(pose_t, ids_t)
            u._agent.write_root_com_velocity_to_sim(
                torch.zeros((1, 6), device=u.device), ids_t)
            return True
    return False


def _refill_label_pool(u):
    """Top up env's visibility_label_pool with another 50/25/25 batch."""
    n = max(2000, u.num_envs * 4)
    a = n // 2
    b = n // 4
    c = n - a - b
    new_labels = (["in_view"] * a + ["occluded"] * b + ["outside_fov"] * c)
    random.shuffle(new_labels)
    u.visibility_label_pool.extend(new_labels)
    print(f"[POOL] refilled +{n} labels "
          f"(in_view={a} occluded={b} outside_fov={c})")


def _advance_slot(u, slot_idx):
    """Manually replenish slot: bump folder_idx + pull next label from pool.

    Done in lieu of `_replenish_slots` because that one short-circuits when
    `next_env_id >= total_envs_to_sim`. We refill the pool here instead.
    """
    if not u.visibility_label_pool:
        _refill_label_pool(u)
    new_env = u.next_env_id
    u.next_env_id += 1
    u.slot_to_env_id[slot_idx] = new_env
    u.slot_folder_indices[slot_idx] = u.next_env_folder_idx + new_env
    u.slot_attempt_counts[slot_idx] = 0
    u.slot_visibility_categories[slot_idx] = u._assign_next_visibility_label(
        u.slot_folder_indices[slot_idx])


# ── per-env episode state ─────────────────────────────────────────────────────

class Ep:
    __slots__ = ("queue", "plan_attempts", "total_steps", "done", "success",
                 "failed", "fail_reason", "env_state", "align_dir",
                 "align_steps", "best_yaw_err", "align_no_progress",
                 "pre_align_yaw", "pre_step_pos", "actions_taken",
                 "folder_idx", "label", "reason")

    def reset(self):
        self.queue           = []
        self.plan_attempts   = 0
        self.total_steps     = 0
        self.done            = False
        self.success         = False
        self.failed          = False
        self.fail_reason     = ""
        self.env_state       = NAVIGATING
        self.align_dir       = None
        self.align_steps     = 0
        self.best_yaw_err    = None
        self.align_no_progress = 0
        self.pre_align_yaw   = None
        self.pre_step_pos    = None
        self.actions_taken   = []
        self.folder_idx      = -1
        self.label           = ""
        self.reason          = ""

    def __init__(self):
        self.reset()


# ── planning ──────────────────────────────────────────────────────────────────

def try_plan(u, i, ep, args):
    _, pre_pos, pre_yaw = u._is_goal_reached_3act(
        env_id=i, pos_tol_m=args.pos_tol, yaw_tol_deg=args.yaw_tol_deg)
    if pre_pos <= args.pos_tol:
        ep.done = True; ep.success = True; return
    if ep.plan_attempts >= args.max_plan_attempts:
        ep.done = True; ep.failed = True
        ep.fail_reason = "max_plan_attempts"; return
    if ep.total_steps >= args.max_total_steps:
        ep.done = True; ep.failed = True
        ep.fail_reason = "max_total_steps_before_plan"; return

    default_infl = float(getattr(u, "planner_inflation_m", 0.12))
    schedule = [None, default_infl * 0.5, default_infl * 0.25, 0.0]
    plan = None
    for infl in schedule:
        if ep.plan_attempts >= args.max_plan_attempts:
            break
        plan = u.plan_to_camera_actions_3act(
            env_id=i, pos_tol_m=args.pos_tol, yaw_tol_deg=args.yaw_tol_deg,
            max_steps=args.max_plan_steps, inflation_radius=infl)
        ep.plan_attempts += 1
        lbl = f"{infl:.3f}" if infl is not None else f"{default_infl:.3f}(default)"
        if plan.get("success", False):
            break
        m = plan.get("metrics", {}) or {}
        print(f"[PLAN-FAIL] env={i} attempt={ep.plan_attempts} "
              f"reason={plan.get('reason','?')} inflation={lbl} "
              f"expanded={m.get('expanded_nodes','?')} pre_pos={pre_pos:.3f}")

    if not plan.get("success", False):
        ep.done = True; ep.failed = True
        ep.fail_reason = f"plan_no_path_{ep.plan_attempts}"; return

    remaining = args.max_total_steps - ep.total_steps
    ep.queue = [int(a) for a in list(plan.get("actions", []))[:remaining]
                if a in (0, 2, 3)]
    if not ep.queue:
        ep.done = True; ep.failed = True
        ep.fail_reason = f"empty_plan_{ep.plan_attempts}"


# ── tracker JSON ─────────────────────────────────────────────────────────────

class SuccessTracker:
    """Persistent JSON of successful episodes; flushed on every save."""

    def __init__(self, path):
        self.path = path
        self.data = {"total": 0,
                     "by_reason": {"in_view": 0, "occluded": 0, "outside_fov": 0},
                     "envs": []}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except Exception:
                pass
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def add(self, entry):
        self.data["total"] += 1
        r = entry.get("reason", "unknown")
        if r in self.data["by_reason"]:
            self.data["by_reason"][r] += 1
        self.data["envs"].append(entry)
        # Atomic flush
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def saved_counts(self):
        return dict(self.data["by_reason"])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=args_cli.seed)
    u = env.unwrapped
    u._disable_auto_reset = True

    # Bump simulation cap so RL-reset replenishment never blocks
    u.total_envs_to_sim = max(u.total_envs_to_sim,
                              args_cli.target_successes * 10)

    print(f"[INFO] base_path={u.base_path}")
    print(f"[INFO] num_envs={args_cli.num_envs} seed={args_cli.seed} "
          f"target={args_cli.target_successes}")

    N        = args_cli.num_envs
    quotas   = {c: int(f * args_cli.target_successes)
                for c, f in TARGET_FRACTIONS.items()}
    tracker  = SuccessTracker(os.path.join(u.base_path,
                                            "successful_envs.json"))
    saved    = tracker.saved_counts()
    for c in TARGET_FRACTIONS:
        saved.setdefault(c, 0)

    eps = [Ep() for _ in range(N)]

    def _quota_met():
        return all(saved[c] >= quotas[c] for c in quotas)

    def _begin_episode(i):
        eps[i].reset()
        eps[i].folder_idx = u.slot_folder_indices[i]
        eps[i].reason     = _reason_for(u, i)
        eps[i].label      = _label_for(u, i)

    def _do_rl_reset(i):
        act = torch.full((N,), NOOP_ACTION, dtype=torch.long, device=u.device)
        act[i] = 6
        env.step(act)
        if not _spawn_agent_near_camera(u, i, rng,
                                        args_cli.spawn_half_extent,
                                        args_cli.spawn_deadzone):
            print(f"[WARN] env={i} spawn-zone failed; using reset position.")
        # Advance to fresh folder_idx + new label from pool
        _advance_slot(u, i)
        _begin_episode(i)

    def _commit_success(i, pos_err, yaw_err):
        ep = eps[i]
        # Save final agent + cam-obj semantic POV via env method
        u.save_rollout_final(i, ep.folder_idx, pos_err, yaw_err)

        # Write actions.txt + meta.json into cam/<label>/rollout/env_*
        dirs = _ep_dirs(u, ep.folder_idx, ep.label)
        os.makedirs(dirs["cam"], exist_ok=True)
        with open(os.path.join(dirs["cam"], "actions.txt"), "w") as f:
            f.write(" ".join(str(a) for a in ep.actions_taken) + "\n")
        meta = {"folder_idx": ep.folder_idx,
                "env_slot": i,
                "seed": args_cli.seed,
                "reason": ep.reason,
                "label": ep.label,
                "total_steps": ep.total_steps,
                "align_steps": ep.align_steps,
                "plan_attempts": ep.plan_attempts,
                "final_pos_err_m": pos_err,
                "final_yaw_err_deg": yaw_err,
                "timestamp": datetime.utcnow().isoformat() + "Z"}
        with open(os.path.join(dirs["cam"], "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Update tracker (flush to disk immediately)
        tracker.add(meta)
        saved[ep.reason] = saved.get(ep.reason, 0) + 1
        print(f"[SAVED] folder_idx={ep.folder_idx} env={i} reason={ep.reason} "
              f"label={ep.label} | {saved}")

    def _discard(i):
        ep = eps[i]
        if ep.folder_idx < 0 or ep.label not in ("Yes", "No"):
            return
        dirs = _ep_dirs(u, ep.folder_idx, ep.label)
        for d in dirs.values():
            shutil.rmtree(d, ignore_errors=True)

    # Initial setup
    for i in range(N):
        _begin_episode(i)
        try_plan(u, i, eps[i], args_cli)

    sim_steps = 0

    while not _quota_met():
        actions          = [NOOP_ACTION] * N
        active_this_step = [False] * N

        # ── pre-step ──────────────────────────────────────────────────────
        for i in range(N):
            ep = eps[i]
            if ep.done:
                continue
            if ep.total_steps >= args_cli.max_total_steps:
                ep.done = True; ep.failed = True
                ep.fail_reason = "max_total_steps"; ep.env_state = DONE_STATE
                continue

            if ep.env_state == NAVIGATING:
                if ep.queue:
                    actions[i] = ep.queue.pop(0)
                    active_this_step[i] = True
                    if actions[i] == 0:
                        ep.pre_step_pos = u._agent.data.root_pos_w[
                            i, :2].cpu().clone()

            elif ep.env_state == ALIGNING_YAW:
                target_yaw = u._get_camera_corrected_yaw(i)
                agent_yaw  = u._get_agent_yaw(i)
                delta      = u._normalize_angle(target_yaw - agent_yaw)
                delta_deg  = math.degrees(abs(delta))

                if abs(delta) < u.heading_bin_rad / 2.0:
                    pos_err_now = float(torch.norm(
                        u._agent.data.root_pos_w[i, :2] -
                        u._camera_obj.data.root_pos_w[i, :2]).item())
                    ep.done    = True
                    ep.success = (pos_err_now <= args_cli.pos_tol and
                                  delta_deg   <= args_cli.yaw_tol_deg)
                    ep.env_state = DONE_STATE
                    print(f"[ALIGN-DONE] env={i} yaw_err={delta_deg:.1f}°")
                    continue

                actions[i]          = 2 if delta > 0 else 3
                active_this_step[i] = True
                if ep.align_dir is None:
                    pe = float(torch.norm(
                        u._agent.data.root_pos_w[i, :2] -
                        u._camera_obj.data.root_pos_w[i, :2]).item())
                    print(f"[ALIGN-START] env={i} pos_err={pe:.3f}m "
                          f"yaw_err={delta_deg:.1f}°")
                ep.align_dir       = actions[i]
                ep.pre_align_yaw   = u._get_agent_yaw(i)

        action_tensor = torch.tensor(actions, dtype=torch.long, device=u.device)
        env.step(action_tensor)
        sim_steps += 1

        # ── post-step: per-frame save + revert + alignment ────────────────
        for i in range(N):
            ep = eps[i]

            if active_this_step[i]:
                ep.total_steps += 1
                ep.actions_taken.append(actions[i])
                # Save RGB + semantic via env method
                try:
                    u.save_rollout_step(i, ep.folder_idx, ep.total_steps,
                                        ACTION_LABELS.get(actions[i],
                                                          f"a{actions[i]}"))
                except Exception as e:
                    print(f"[WARN] save_rollout_step env={i} failed: {e}")

            # Forward-revert detection
            if (ep.env_state == NAVIGATING and actions[i] == 0
                    and ep.pre_step_pos is not None):
                disp = float(torch.norm(
                    u._agent.data.root_pos_w[i, :2].cpu() -
                    ep.pre_step_pos).item())
                if disp < 0.01:
                    ep.queue.clear()
                    print(f"[REVERT] env={i} step={ep.total_steps} "
                          f"fwd reverted ({disp:.4f}m). re-plan.")
                ep.pre_step_pos = None

            # Yaw alignment post-step
            if ep.env_state != ALIGNING_YAW or not active_this_step[i]:
                continue

            ep.align_steps += 1
            yaw_err_now = _compute_yaw_err(u, i)
            pos_err_now = float(torch.norm(
                u._agent.data.root_pos_w[i, :2] -
                u._camera_obj.data.root_pos_w[i, :2]).item())

            if ep.pre_align_yaw is not None:
                yd = abs(u._normalize_angle(
                    u._get_agent_yaw(i) - ep.pre_align_yaw))
                ep.align_no_progress = (ep.align_no_progress + 1
                                        if yd < u.heading_bin_rad * 0.25
                                        else 0)
            ep.pre_align_yaw = None

            if ep.align_no_progress >= 4:
                ep.done = True; ep.env_state = DONE_STATE
                print(f"[ALIGN-STUCK] env={i} yaw_err={yaw_err_now:.1f}°")
                continue

            new_delta = u._normalize_angle(
                u._get_camera_corrected_yaw(i) - u._get_agent_yaw(i))
            new_dir = 2 if new_delta > 0 else 3

            if new_dir != actions[i]:
                ep.done    = True
                ep.success = (pos_err_now <= args_cli.pos_tol and
                              yaw_err_now  <= args_cli.yaw_tol_deg)
                ep.env_state = DONE_STATE
                print(f"[ALIGN-STOP] env={i} overshoot "
                      f"yaw_err={yaw_err_now:.1f}° steps={ep.align_steps}")
                continue

            if ep.best_yaw_err is None or yaw_err_now < ep.best_yaw_err:
                ep.best_yaw_err = yaw_err_now

            if ep.align_steps >= args_cli.max_align_steps:
                ep.done    = True
                ep.success = (pos_err_now <= args_cli.pos_tol and
                              yaw_err_now  <= args_cli.yaw_tol_deg)
                ep.env_state = DONE_STATE
                print(f"[ALIGN-MAX] env={i} yaw_err={yaw_err_now:.1f}°")

        # ── post-step: queue-drained transitions ──────────────────────────
        for i in range(N):
            ep = eps[i]
            if ep.done or ep.env_state != NAVIGATING or ep.queue:
                continue

            _, pos_err, yaw_err = u._is_goal_reached_3act(
                env_id=i, pos_tol_m=args_cli.pos_tol,
                yaw_tol_deg=args_cli.yaw_tol_deg)

            if pos_err <= args_cli.pos_tol:
                if yaw_err <= args_cli.yaw_tol_deg:
                    ep.done = True; ep.success = True; ep.env_state = DONE_STATE
                    print(f"[ALIGN-SKIP] env={i} yaw_err={yaw_err:.1f}°")
                    continue
                ep.env_state         = ALIGNING_YAW
                ep.align_dir         = None
                ep.align_steps       = 0
                ep.best_yaw_err      = yaw_err
                ep.align_no_progress = 0
                print(f"[ALIGN-START] env={i} pos_err={pos_err:.3f}m "
                      f"yaw_err={yaw_err:.1f}°")
                continue

            if ep.total_steps >= args_cli.max_total_steps:
                ep.done = True; ep.failed = True
                ep.fail_reason = "max_total_steps_after_exec"
                ep.env_state = DONE_STATE; continue
            if ep.plan_attempts >= args_cli.max_plan_attempts:
                ep.done = True; ep.failed = True
                ep.fail_reason = "max_plan_attempts_after_exec"
                ep.env_state = DONE_STATE; continue

            try_plan(u, i, ep, args_cli)
            if not ep.queue and not ep.done:
                ep.done = True; ep.failed = True
                ep.fail_reason = "empty_replan"; ep.env_state = DONE_STATE

        # ── episode finalization ──────────────────────────────────────────
        for i in range(N):
            ep = eps[i]
            if not ep.done:
                continue

            # Recompute final errors for record
            _, pos_err, yaw_err = u._is_goal_reached_3act(
                env_id=i, pos_tol_m=args_cli.pos_tol,
                yaw_tol_deg=args_cli.yaw_tol_deg)

            within_quota = saved.get(ep.reason, 0) < quotas.get(ep.reason, 0)

            if ep.success and within_quota:
                _commit_success(i, pos_err, yaw_err)
                print(f"[DONE] env={i} reason={ep.reason} SUCCESS "
                      f"steps={ep.total_steps} align={ep.align_steps}")
            else:
                tag = "OVERFLOW" if ep.success else "FAIL"
                print(f"[DONE] env={i} reason={ep.reason} {tag} "
                      f"steps={ep.total_steps} failed={ep.failed} "
                      f"fail_reason={ep.fail_reason}")
                _discard(i)

            if _quota_met():
                break

            _do_rl_reset(i)
            try_plan(u, i, eps[i], args_cli)

    print(f"\n[DONE] Collection complete. {saved} (target {quotas}) "
          f"in {sim_steps} sim steps.")
    print(f"[INFO] Tracker JSON: "
          f"{os.path.join(u.base_path, 'successful_envs.json')}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
