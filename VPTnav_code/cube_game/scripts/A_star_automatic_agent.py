# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Automatic A* rollout driver.

Plans + executes A* navigation to the camera object for N parallel envs.

Per-env protocol:
 1. Plan A* path.
 2. Execute the planned action sequence step-by-step.
 3. When queue drains and position reached: greedy yaw alignment.
    Compute delta = normalize(target_yaw - agent_yaw).
    delta > 0 → turn left (action 2); delta < 0 → turn right (action 3).
    Keep stepping until yaw error increases (overshoot) or within tol.
    Save final-pose image.

Caps per env:
 - Max 4 plan attempts (with inflation fallback).
 - Max 100 total executed sim steps.

Saves per env to `{env.base_path}/rollout/env_{idx}/`:
 - `step_{N:05d}_{label}.png` per executed step (agent POV from `env.obs`).
 - `step_{N:05d}_env_{idx:04d}_final_pose_{pos_err:.3f}m_yaw_{yaw_err:.1f}d.png`
   when alignment finishes.
 - `seed_{seed}_env_{idx}_{reason}[_failed].json` with metrics + trajectory.

`reason` is read from `env.env_visibility_reasons[folder_idx]`
(in_view / occluded / outside_fov).
"""

import argparse
import json
import os
import signal
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher


def force_exit(signum, frame):
    print(f"\nForce killing self (PID: {os.getpid()})...")
    os.kill(os.getpid(), signal.SIGKILL)


signal.signal(signal.SIGINT, force_exit)
signal.signal(signal.SIGTSTP, force_exit)

parser = argparse.ArgumentParser(description="Automatic A* rollout driver.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="VPT-v18-A-star")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--pos_tol",
    type=float,
    default=0.3,
    help="Distance tol in meters for success (matches env termination).")
parser.add_argument(
    "--yaw_tol_deg",
    type=float,
    default=15,
    help="Yaw tol in degrees for success (= 0.2 rad, matches env termination)."
)
parser.add_argument("--max_plan_steps",
    type=int,
    default=512,
    help="Max action horizon a single A* plan may produce.")
parser.add_argument("--max_plan_attempts",
    type=int,
    default=4,
    help="Max times we re-plan per env before giving up.")
parser.add_argument("--max_total_steps",
    type=int,
    default=200,
    help="Max executed sim steps per env across all plans.")
parser.add_argument("--max_align_steps",
    type=int,
    default=50,
    help="Max yaw alignment steps after position reached.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import cube_game.tasks  # noqa: F401

import math

import cv2
import gymnasium as gym
import torch

# Sim-level no-op: any action not in {0,1,2,3,5,6} falls through `move_agent`
# without touching pose. Used for done/idle envs in the action vector.
NOOP_ACTION = 4
ACTION_LABELS = {0: "fwd", 1: "back", 2: "left", 3: "right", 4: "noop"}

# Per-env execution states.
NAVIGATING = 0      # executing A* action queue
ALIGNING_YAW = 1    # position reached, greedy yaw alignment
DONE_STATE = 2      # finished (success or failure)


def _compute_yaw_err(env_unwrapped, env_idx: int) -> float:
    """Return absolute yaw error in degrees for env_idx."""
    target_yaw = env_unwrapped._get_camera_corrected_yaw(env_idx)
    agent_yaw = env_unwrapped._get_agent_yaw(env_idx)
    return abs(math.degrees(
        env_unwrapped._normalize_angle(target_yaw - agent_yaw)))


def _save_env_frame(env_unwrapped, env_idx: int, rollout_root: str,
                    step_n: int, action_label: str):
    """Save the env_idx camera POV frame from `env.obs` to disk."""
    if not hasattr(env_unwrapped, "obs"):
        return
    obs = env_unwrapped.obs
    if obs is None or env_idx >= obs.shape[0]:
        return
    frame = obs[env_idx].permute(1, 2, 0).contiguous()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    if frame.dtype != "uint8":
        frame = frame.astype("uint8")
    env_dir = os.path.join(rollout_root, f"env_{env_idx}")
    os.makedirs(env_dir, exist_ok=True)
    fname = f"step_{step_n:05d}_{action_label}.png"
    cv2.imwrite(os.path.join(env_dir, fname),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _save_final_pose_frame(env_unwrapped, env_idx: int, rollout_root: str,
                           step_n: int, pos_err: float, yaw_err: float):
    """Save a tagged final-pose image with both pos and yaw error."""
    if not hasattr(env_unwrapped, "obs"):
        return
    obs = env_unwrapped.obs
    if obs is None or env_idx >= obs.shape[0]:
        return
    frame = obs[env_idx].permute(1, 2, 0).contiguous()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    if frame.dtype != "uint8":
        frame = frame.astype("uint8")
    env_dir = os.path.join(rollout_root, f"env_{env_idx}")
    os.makedirs(env_dir, exist_ok=True)
    tag = f"env_{env_idx:04d}_final_pose_{pos_err:.3f}m_yaw_{yaw_err:.1f}d"
    fname = f"step_{step_n:05d}_{tag}.png"
    cv2.imwrite(os.path.join(env_dir, fname),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"[FINAL-POSE] env={env_idx} saved {fname}")


def _read_pose(env_unwrapped, env_idx: int):
    """Return (x, y, yaw) for env_idx in world frame."""
    pos = env_unwrapped._agent.data.root_pos_w[
        env_idx, :2].cpu().numpy().tolist()
    yaw = float(env_unwrapped._get_agent_yaw(env_idx))
    return [float(pos[0]), float(pos[1]), yaw]


def _read_target(env_unwrapped, env_idx: int):
    cam = env_unwrapped._camera_obj.data.root_pos_w[
        env_idx, :2].cpu().numpy().tolist()
    goal = env_unwrapped._goal.data.root_pos_w[
        env_idx, :2].cpu().numpy().tolist()
    return [float(cam[0]), float(cam[1])], [float(goal[0]), float(goal[1])]


def _reason_for(env_unwrapped, env_idx: int) -> str:
    folder_idx = env_unwrapped.slot_folder_indices[env_idx]
    return env_unwrapped.env_visibility_reasons.get(folder_idx, "unknown")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    u = env.unwrapped
    # Disable auto-reset so finishing one env doesn't scramble the scene
    # for the others mid-rollout.
    u._disable_auto_reset = True

    rollout_root = os.path.join(u.base_path, "rollout")
    os.makedirs(rollout_root, exist_ok=True)
    print(f"[INFO] Rollout root: {rollout_root}")
    print(f"[INFO] num_envs={args_cli.num_envs} seed={args_cli.seed}")
    print(f"[INFO] max_plan_attempts={args_cli.max_plan_attempts} "
          f"max_total_steps={args_cli.max_total_steps}")
    print(f"[INFO] Yaw alignment: delta-based direction pick, "
          f"stop on overshoot. max_align_steps={args_cli.max_align_steps}")

    # Sanity-check tolerances against the underlying sim grid:
    forward_step_m = float(getattr(u, "forward_step_m", 0.15))
    heading_bin_deg = float(
        math.degrees(getattr(u, "heading_bin_rad", math.pi / 24)))
    pos_tol_lower = forward_step_m / 2.0
    yaw_tol_lower = heading_bin_deg / 2.0
    if args_cli.pos_tol < pos_tol_lower:
        print(
            f"[WARN] pos_tol={args_cli.pos_tol} is below sim resolution lower bound "
            f"{pos_tol_lower:.3f} m (= forward_step / 2). Plans may fail spuriously."
        )
    if args_cli.yaw_tol_deg < yaw_tol_lower:
        print(
            f"[WARN] yaw_tol_deg={args_cli.yaw_tol_deg} is below sim resolution lower bound "
            f"{yaw_tol_lower:.2f} deg (= heading_bin / 2). Plans may fail spuriously."
        )

    N = args_cli.num_envs
    queues: list[list[int]] = [[] for _ in range(N)]
    plan_attempts = [0] * N
    total_steps = [0] * N
    done = [False] * N
    success = [False] * N
    failed = [False] * N
    fail_reason = [""] * N

    # Execution state: NAVIGATING, ALIGNING_YAW, or DONE_STATE.
    env_state = [NAVIGATING] * N

    # Greedy yaw alignment bookkeeping per env.
    align_dir = [None] * N          # 2=left, 3=right
    align_steps = [0] * N           # steps spent aligning
    best_yaw_err = [None] * N       # best (lowest) yaw err seen during alignment
    align_no_progress = [0] * N     # consecutive steps with no yaw improvement
    pre_align_yaw = [None] * N      # yaw before each turn step (turn-revert detection)

    # Per-env logs.
    plan_log: list[list[dict]] = [[] for _ in range(N)]
    traj_log: list[list[list[float]]] = [[_read_pose(u, i)] for i in range(N)]
    # Record every action executed per env.
    actions_taken_log: list[list[int]] = [[] for _ in range(N)]
    init_cam = []
    init_goal = []
    init_reason = []
    init_pose = []
    init_pos_err = [0.0] * N
    init_yaw_err = [0.0] * N
    for i in range(N):
        cam, goal = _read_target(u, i)
        init_cam.append(cam)
        init_goal.append(goal)
        init_reason.append(_reason_for(u, i))
        init_pose.append(_read_pose(u, i))
        _, p_err, y_err = u._is_goal_reached_3act(
            env_id=i,
            pos_tol_m=args_cli.pos_tol,
            yaw_tol_deg=args_cli.yaw_tol_deg,
        )
        init_pos_err[i] = p_err
        init_yaw_err[i] = y_err
        # Save the spawn frame as step_00000 so failed envs always have at
        # least one image even if no actions were ever executed.
        _save_env_frame(u, i, rollout_root, step_n=0, action_label="spawn")

    def needs_plan(i: int) -> bool:
        return (not done[i]) and (not queues[i])

    def try_plan_env(i: int):
        """Plan for env i. Updates queue + bookkeeping in place.

        On ``no_path``, retries with progressively halved inflation
        radius before giving up.
        """
        reached_pre, pre_pos, pre_yaw = u._is_goal_reached_3act(
            env_id=i,
            pos_tol_m=args_cli.pos_tol,
            yaw_tol_deg=args_cli.yaw_tol_deg,
        )
        if pre_pos <= args_cli.pos_tol:
            done[i] = True
            success[i] = True
            return
        if plan_attempts[i] >= args_cli.max_plan_attempts:
            done[i] = True
            failed[i] = True
            fail_reason[i] = "max_plan_attempts"
            return
        if total_steps[i] >= args_cli.max_total_steps:
            done[i] = True
            failed[i] = True
            fail_reason[i] = "max_total_steps_before_plan"
            return

        # Inflation fallback schedule: default, then 0.5x, 0.25x, 0.
        default_inflation = float(getattr(u, "planner_inflation_m", 0.12))
        inflation_schedule = [
            None,  # None = use default from env
            default_inflation * 0.5,
            default_inflation * 0.25,
            0.0,
        ]

        plan = None
        for infl in inflation_schedule:
            if plan_attempts[i] >= args_cli.max_plan_attempts:
                break
            plan = u.plan_to_camera_actions_3act(
                env_id=i,
                pos_tol_m=args_cli.pos_tol,
                yaw_tol_deg=args_cli.yaw_tol_deg,
                max_steps=args_cli.max_plan_steps,
                inflation_radius=infl,
            )
            plan_attempts[i] += 1
            infl_label = (f"{infl:.3f}" if infl is not None else
                          f"{default_inflation:.3f}(default)")
            plan_log[i].append({
                "attempt": plan_attempts[i],
                "success": bool(plan.get("success", False)),
                "reason": plan.get("reason", "unknown"),
                "actions_len": len(plan.get("actions", [])),
                "metrics": plan.get("metrics", {}),
                "pre_pos_err": pre_pos,
                "pre_yaw_err_deg": pre_yaw,
                "inflation_m": infl_label,
            })
            if plan.get("success", False):
                break
            m = plan.get("metrics", {}) or {}
            print(f"[PLAN-FAIL] env={i} attempt={plan_attempts[i]} "
                  f"reason={plan.get('reason','?')} "
                  f"inflation={infl_label} "
                  f"expanded={m.get('expanded_nodes','?')} "
                  f"pre_pos={pre_pos:.3f} pre_yaw={pre_yaw:.2f}")

        if not plan.get("success", False):
            done[i] = True
            failed[i] = True
            fail_reason[i] = f"plan_no_path_attempt_{plan_attempts[i]}"
            return
        # Cap remaining steps so we don't blow the per-env budget.
        remaining = args_cli.max_total_steps - total_steps[i]
        actions = list(plan.get("actions", []))[:remaining]
        # Filter to executable actions only ({0,2,3}); skip planner noise.
        queues[i] = [int(a) for a in actions if a in (0, 2, 3)]
        if not queues[i]:
            done[i] = True
            failed[i] = True
            fail_reason[
                i] = f"empty_actionable_plan_attempt_{plan_attempts[i]}"

    # Initial plan for all envs.
    for i in range(N):
        if needs_plan(i):
            try_plan_env(i)

    sim_step_idx = 0
    # Pre-step positions for collision-revert detection.
    pre_step_pos = [None] * N
    while not all(done):
        actions = [NOOP_ACTION] * N
        active_this_step = [False] * N
        for i in range(N):
            if done[i]:
                continue
            if total_steps[i] >= args_cli.max_total_steps:
                done[i] = True
                failed[i] = True
                fail_reason[i] = "max_total_steps"
                env_state[i] = DONE_STATE
                continue

            if env_state[i] == NAVIGATING:
                if queues[i]:
                    actions[i] = queues[i].pop(0)
                    active_this_step[i] = True
                    if actions[i] == 0:
                        pre_step_pos[i] = u._agent.data.root_pos_w[
                            i, :2].cpu().clone()
                # else: queue empty -- handled post-step below

            elif env_state[i] == ALIGNING_YAW:
                # Dynamic greedy yaw alignment for the BEST fit:
                # Re-calculate delta and pick best direction at every step.
                target_yaw = u._get_camera_corrected_yaw(i)
                agent_yaw = u._get_agent_yaw(i)
                delta = u._normalize_angle(target_yaw - agent_yaw)
                delta_deg = math.degrees(abs(delta))

                # GREEDY STOP: If we are already closer than half a step,
                # any further move will increase the error.
                if abs(delta) < (u.heading_bin_rad / 2.0):
                    done[i] = True
                    env_state[i] = DONE_STATE
                    pos_err_now = float(torch.norm(
                        u._agent.data.root_pos_w[i, :2] -
                        u._camera_obj.data.root_pos_w[i, :2]).item())
                    success[i] = (pos_err_now <= args_cli.pos_tol) and (delta_deg <= args_cli.yaw_tol_deg)
                    _save_final_pose_frame(u, i, rollout_root,
                                           step_n=total_steps[i] + 1,
                                           pos_err=pos_err_now,
                                           yaw_err=delta_deg)
                    print(f"[ALIGN-DONE] env={i} best_fit achieved "
                          f"yaw_err={delta_deg:.1f}°")
                    continue

                if delta > 0:
                    actions[i] = 2  # left
                else:
                    actions[i] = 3  # right

                if align_dir[i] is None:
                    print(f"[ALIGN-START] env={i} yaw_err={delta_deg:.1f}°")

                align_dir[i] = actions[i]
                active_this_step[i] = True
                pre_align_yaw[i] = u._get_agent_yaw(i)

        action_tensor = torch.tensor(actions,
                                     dtype=torch.long,
                                     device=u.device)
        env.step(action_tensor)
        sim_step_idx += 1

        for i in range(N):
            if active_this_step[i]:
                total_steps[i] += 1
                actions_taken_log[i].append(actions[i])
                label = ACTION_LABELS.get(actions[i], f"a{actions[i]}")
                _save_env_frame(u, i, rollout_root, total_steps[i], label)
                traj_log[i].append(_read_pose(u, i))

            # Collision-revert detection for NAVIGATING state:
            if (env_state[i] == NAVIGATING and actions[i] == 0
                    and pre_step_pos[i] is not None):
                post_pos = u._agent.data.root_pos_w[i, :2].cpu()
                displacement = float(
                    torch.norm(post_pos - pre_step_pos[i]).item())
                if displacement < 0.01:  # < 1 cm means collision revert
                    queues[i].clear()
                    print(
                        f"[REVERT] env={i} step={total_steps[i]} "
                        f"fwd action reverted (disp={displacement:.4f}m). "
                        f"Clearing queue for re-plan.")
                    pre_step_pos[i] = None

            # ── Greedy yaw alignment post-step ──────────────────────
            if env_state[i] == ALIGNING_YAW and active_this_step[i]:
                align_steps[i] += 1
                yaw_err_now = _compute_yaw_err(u, i)
                pos_err_now = float(torch.norm(
                    u._agent.data.root_pos_w[i, :2] -
                    u._camera_obj.data.root_pos_w[i, :2]).item())

                # Turn-revert detection: if yaw didn't change by at least
                # 25% of one heading bin, the turn was reverted by collision.
                turn_revert_threshold = u.heading_bin_rad * 0.25
                if pre_align_yaw[i] is not None:
                    yaw_after = u._get_agent_yaw(i)
                    yaw_delta = abs(u._normalize_angle(yaw_after - pre_align_yaw[i]))
                    if yaw_delta < turn_revert_threshold:
                        align_no_progress[i] += 1
                        print(f"[ALIGN-REVERT] env={i} turn reverted "
                              f"(delta={math.degrees(yaw_delta):.2f}°) "
                              f"no_progress={align_no_progress[i]}")
                    else:
                        align_no_progress[i] = 0
                pre_align_yaw[i] = None

                # Stuck detection: 4 consecutive steps with no yaw progress.
                if align_no_progress[i] >= 4:
                    done[i] = True
                    env_state[i] = DONE_STATE
                    _save_final_pose_frame(u, i, rollout_root,
                                           step_n=total_steps[i] + 1,
                                           pos_err=pos_err_now,
                                           yaw_err=yaw_err_now)
                    print(f"[ALIGN-STUCK] env={i} cannot turn further. "
                          f"yaw_err={yaw_err_now:.1f}°")
                    continue

                # Check for overshoot: if the optimal direction now is OPPOSITE
                # to the action we just took, we crossed the target.
                target_yaw = u._get_camera_corrected_yaw(i)
                agent_yaw = u._get_agent_yaw(i)
                new_delta = u._normalize_angle(target_yaw - agent_yaw)
                new_dir = 2 if new_delta > 0 else 3

                if new_dir != actions[i]:
                    # Direction flipped → overshoot. Success only if within tol.
                    done[i] = True
                    success[i] = (pos_err_now <= args_cli.pos_tol and
                                  yaw_err_now <= args_cli.yaw_tol_deg)
                    env_state[i] = DONE_STATE
                    _save_final_pose_frame(u, i, rollout_root,
                                           step_n=total_steps[i] + 1,
                                           pos_err=pos_err_now,
                                           yaw_err=yaw_err_now)
                    print(f"[ALIGN-STOP] env={i} overshoot (dir flipped) "
                          f"yaw_err={yaw_err_now:.1f}° align_steps={align_steps[i]}")
                    continue

                # Update running best.
                if best_yaw_err[i] is None or yaw_err_now < best_yaw_err[i]:
                    best_yaw_err[i] = yaw_err_now

                # Hit max alignment steps → stop.
                if align_steps[i] >= args_cli.max_align_steps:
                    done[i] = True
                    success[i] = (pos_err_now <= args_cli.pos_tol and
                                  yaw_err_now <= args_cli.yaw_tol_deg)
                    env_state[i] = DONE_STATE
                    _save_final_pose_frame(u, i, rollout_root,
                                           step_n=total_steps[i] + 1,
                                           pos_err=pos_err_now,
                                           yaw_err=yaw_err_now)
                    print(f"[ALIGN-MAX] env={i} hit max_align_steps="
                          f"{args_cli.max_align_steps} "
                          f"yaw_err={yaw_err_now:.1f}°")
                    continue

        # Post-step: handle state transitions for NAVIGATING envs
        # whose queues just became empty.
        for i in range(N):
            if done[i]:
                continue
            if env_state[i] != NAVIGATING:
                continue
            if queues[i]:
                continue

            # Queue just drained. Check position.
            _, pos_err, yaw_err = u._is_goal_reached_3act(
                env_id=i,
                pos_tol_m=args_cli.pos_tol,
                yaw_tol_deg=args_cli.yaw_tol_deg,
            )
            # Position reached → start yaw alignment.
            if pos_err <= args_cli.pos_tol:
                # Already within yaw tol → done immediately.
                if yaw_err <= args_cli.yaw_tol_deg:
                    done[i] = True
                    success[i] = True
                    env_state[i] = DONE_STATE
                    _save_final_pose_frame(u, i, rollout_root,
                                           step_n=total_steps[i] + 1,
                                           pos_err=pos_err,
                                           yaw_err=yaw_err)
                    print(f"[ALIGN-SKIP] env={i} already aligned "
                          f"yaw_err={yaw_err:.1f}° ≤ tol={args_cli.yaw_tol_deg}°")
                    continue
                env_state[i] = ALIGNING_YAW
                align_dir[i] = None      # will be set from delta
                align_steps[i] = 0
                best_yaw_err[i] = yaw_err  # baseline = pre-alignment error
                print(f"[ALIGN-START] env={i} pos_err={pos_err:.3f}m "
                      f"yaw_err={yaw_err:.1f}°")
                continue

            # Position not reached. Try to re-plan.
            if total_steps[i] >= args_cli.max_total_steps:
                done[i] = True
                failed[i] = True
                fail_reason[i] = "max_total_steps_after_exec"
                env_state[i] = DONE_STATE
                continue
            if plan_attempts[i] >= args_cli.max_plan_attempts:
                done[i] = True
                failed[i] = True
                fail_reason[i] = "max_plan_attempts_after_exec"
                env_state[i] = DONE_STATE
                continue
            try_plan_env(i)
            # If re-plan gave us nothing, mark failed.
            if not queues[i] and not done[i]:
                done[i] = True
                failed[i] = True
                fail_reason[i] = "empty_replan"
                env_state[i] = DONE_STATE

    # Persist per-env metadata JSON.
    for i in range(N):
        reached_final, pos_err, yaw_err = u._is_goal_reached_3act(
            env_id=i,
            pos_tol_m=args_cli.pos_tol,
            yaw_tol_deg=args_cli.yaw_tol_deg,
        )
        ok = (pos_err <= args_cli.pos_tol) and (yaw_err <= args_cli.yaw_tol_deg)
        if not ok and pos_err > args_cli.pos_tol:
            failed[i] = True
            if not fail_reason[i]:
                fail_reason[i] = "tolerance_not_met"
            success[i] = False
        # If the env failed, save a final snapshot.
        if failed[i]:
            _save_env_frame(u,
                            i,
                            rollout_root,
                            step_n=max(total_steps[i] + 1, 1),
                            action_label="final_failed")
        meta = {
            "seed": args_cli.seed,
            "env_idx": i,
            "task": args_cli.task,
            "folder_idx": int(u.slot_folder_indices[i]),
            "reason": init_reason[i],
            "success": bool(success[i] and not failed[i]),
            "failed": bool(failed[i]),
            "fail_reason": fail_reason[i],
            "plan_attempts": plan_attempts[i],
            "total_steps_executed": total_steps[i],
            "pos_tol_m": args_cli.pos_tol,
            "yaw_tol_deg": args_cli.yaw_tol_deg,
            "init_pos_err_m": init_pos_err[i],
            "init_yaw_err_deg": init_yaw_err[i],
            "final_pos_err_m": pos_err,
            "final_yaw_err_deg": yaw_err,
            "align_steps": align_steps[i],
            "best_yaw_err_deg": best_yaw_err[i],
            "init_agent_pose": init_pose[i],
            "init_camera_xy": init_cam[i],
            "init_goal_xy": init_goal[i],
            "trajectory_xyy": traj_log[i],
            "actions_taken": actions_taken_log[i],
            "plans": plan_log[i],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        suffix = "_failed" if failed[i] else ""
        meta_name = (
            f"seed_{args_cli.seed}_env_{i}_{init_reason[i]}{suffix}.json")
        env_dir = os.path.join(rollout_root, f"env_{i}")
        os.makedirs(env_dir, exist_ok=True)
        with open(os.path.join(env_dir, meta_name), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[DONE] env={i} reason={init_reason[i]} "
              f"success={meta['success']} failed={failed[i]} "
              f"plans={plan_attempts[i]} steps={total_steps[i]} "
              f"align={align_steps[i]} "
              f"pos_err={pos_err:.3f} yaw_err={yaw_err:.2f}")

    print(f"[INFO] Total sim steps in this batch: {sim_step_idx}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
