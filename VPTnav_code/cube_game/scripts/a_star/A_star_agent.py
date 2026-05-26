#!/usr/bin/env python3
"""Interactive A* debug agent for step-by-step QC and image capture."""

import argparse
import math
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


parser = argparse.ArgumentParser(description="A* debug agent for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="VPT-v18-A-star", help="Name of the task.")
parser.add_argument("--pos_tol", type=float, default=0.2, help="Planner position tolerance (meters).")
parser.add_argument("--yaw_tol_deg", type=float, default=11.46, help="Planner yaw tolerance (degrees).")
parser.add_argument("--max_plan_steps", type=int, default=512, help="Max action horizon for planner.")
parser.add_argument("--max_plan_attempts", type=int, default=4, help="Max re-plan attempts with inflation fallback.")
parser.add_argument("--save_dir", type=str, default="astar_debug_frames", help="Directory for saved debug frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ── Launch Isaac ────────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac imports (after AppLauncher) ───────────────────────────────────────
import torch
from isaaclab.envs import gym

from cube_game.tasks.direct.cube_game.vpt_env_cfg_v17 import VPTEnvCfg
from cube_game.tasks.direct.cube_game.vpt_env_v18_A_star import VPTEnvAStar


def parse_env_cfg(task_name: str, device: str, num_envs: int, use_fabric: bool):
    cfg = VPTEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.sim.use_fabric = use_fabric
    return cfg


def _save_debug_frame(env, save_dir: str, frame_idx: int, action_label: str):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    rgb = env.unwrapped._rgb_data[0].cpu().numpy()  # (H, W, 3)
    depth = env.unwrapped._depth_data[0].cpu().numpy()  # (H, W)
    import numpy as np
    from PIL import Image
    rgb_path = os.path.join(save_dir, f"{frame_idx:04d}_{action_label}_{ts}_rgb.png")
    depth_path = os.path.join(save_dir, f"{frame_idx:04d}_{action_label}_{ts}_depth.png")
    Image.fromarray(rgb).save(rgb_path)
    depth_vis = np.clip(depth / float(depth.max() + 1e-6), 0, 1)
    Image.fromarray((depth_vis * 255).astype(np.uint8)).save(depth_path)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    env.reset()

    # Disable auto-reset so the scene doesn't re-randomize the moment the
    # agent gets within tolerance of the camera.
    env.unwrapped._disable_auto_reset = True

    u = env.unwrapped

    # ── Navigation state machine ─────────────────────────────────────────
    NAVIGATING = 0
    ALIGNING_YAW = 1
    DONE_STATE = 2

    plan_queue = []
    plan_attempts = 0
    env_state = NAVIGATING
    prev_yaw_err = None
    yaw_increase_count = 0
    MAX_YAW_INCREASES = 3
    total_steps = 0
    frame_idx = 0

    # Inflation fallback schedule (matches A_star_automatic_agent.py)
    default_inflation = float(getattr(u, "planner_inflation_m", 0.12))
    inflation_schedule = [
        None,  # None = use default from env
        default_inflation * 0.5,
        default_inflation * 0.25,
        0.0,
    ]

    print("\n[INFO] A* Debug Controls")
    print("  f : Step forward (action 0) + save image")
    print("  l : Step left turn (action 2) + save image")
    print("  r : Step right turn(action 3) + save image")
    print("  p : Plan A* for env_0 (with inflation fallback)")
    print("  n : Execute NEXT planned action + save image")
    print("  m : Execute all remaining planned actions (step-by-step, saves each)")
    print("  a : Auto-navigate: plan + execute + align until done or failed")
    print("  s : Save current frame only (no step)")
    print("  x : Soft reset (action 5) + save image")
    print("  q : Quit\n")

    # ── Helper: plan with inflation fallback ──────────────────────────────
    def try_plan():
        """Plan with inflation fallback. Returns action list or empty."""
        nonlocal plan_attempts
        if plan_attempts >= args_cli.max_plan_attempts:
            print(f"[PLAN] Max attempts ({args_cli.max_plan_attempts}) reached.")
            return []

        reached_pre, pre_pos, pre_yaw = u._is_goal_reached_3act(
            env_id=0,
            pos_tol_m=args_cli.pos_tol,
            yaw_tol_deg=args_cli.yaw_tol_deg,
        )
        if reached_pre:
            print(f"[PLAN] Already at goal (pos={pre_pos:.3f}m yaw={pre_yaw:.2f}deg)")
            return ["DONE"]

        plan = None
        for infl in inflation_schedule:
            if plan_attempts >= args_cli.max_plan_attempts:
                break
            plan = u.plan_to_camera_actions_3act(
                env_id=0,
                pos_tol_m=args_cli.pos_tol,
                yaw_tol_deg=args_cli.yaw_tol_deg,
                max_steps=args_cli.max_plan_steps,
                inflation_radius=infl,
            )
            plan_attempts += 1
            infl_label = (
                f"{infl:.3f}" if infl is not None
                else f"{default_inflation:.3f}(default)"
            )
            if plan.get("success", False):
                print(
                    f"[PLAN] success=True attempt={plan_attempts} "
                    f"inflation={infl_label} actions={len(plan.get('actions', []))}"
                )
                break
            m = plan.get("metrics", {}) or {}
            print(
                f"[PLAN-FAIL] attempt={plan_attempts} "
                f"reason={plan.get('reason', '?')} inflation={infl_label} "
                f"expanded={m.get('expanded_nodes', '?')}"
            )

        if not plan.get("success", False):
            print("[PLAN] All attempts failed.")
            return []
        actions = list(plan.get("actions", []))
        return [int(a) for a in actions if a in (0, 2, 3)]

    # ── Helper: step + save + displacement ────────────────────────────────
    def step_action(action):
        """Execute one action, save frame, return displacement for collision-revert."""
        nonlocal total_steps, frame_idx
        pre_pos = None
        if action == 0:
            pre_pos = u._agent.data.root_pos_w[0, :2].cpu().clone()
        actions_tensor = torch.full(
            (u.num_envs,), action, dtype=torch.long, device=u.device
        )
        env.step(actions_tensor)
        total_steps += 1
        label = {0: "fwd", 2: "left", 3: "right"}.get(action, f"a{action}")
        _save_debug_frame(
            env, save_dir=args_cli.save_dir, frame_idx=frame_idx, action_label=label
        )
        frame_idx += 1
        displacement = None
        if action == 0 and pre_pos is not None:
            post_pos = u._agent.data.root_pos_w[0, :2].cpu()
            displacement = float(torch.norm(post_pos - pre_pos).item())
        return displacement

    # ── Helper: greedy yaw alignment ──────────────────────────────────────
    def greedy_align_one():
        """Execute one greedy yaw-alignment turn. Returns new yaw error."""
        nonlocal prev_yaw_err, yaw_increase_count, env_state, total_steps, frame_idx
        target_yaw = u._get_camera_corrected_yaw(0)
        agent_yaw = u._get_agent_yaw(0)
        delta = u._normalize_angle(target_yaw - agent_yaw)
        yaw_err = abs(math.degrees(delta))

        if yaw_err <= args_cli.yaw_tol_deg:
            env_state = DONE_STATE
            print(f"[ALIGN] Done! yaw_err={yaw_err:.2f}deg <= tol={args_cli.yaw_tol_deg}")
            return yaw_err

        turn = 2 if delta > 0 else 3
        actions_tensor = torch.full(
            (u.num_envs,), turn, dtype=torch.long, device=u.device
        )
        env.step(actions_tensor)
        total_steps += 1
        label = "align_left" if turn == 2 else "align_right"
        _save_debug_frame(
            env, save_dir=args_cli.save_dir, frame_idx=frame_idx, action_label=label
        )
        frame_idx += 1

        # Check if yaw improved
        new_target = u._get_camera_corrected_yaw(0)
        new_agent = u._get_agent_yaw(0)
        new_yaw_err = abs(math.degrees(u._normalize_angle(new_target - new_agent)))

        if new_yaw_err <= args_cli.yaw_tol_deg:
            env_state = DONE_STATE
            print(f"[ALIGN] Done! yaw_err={new_yaw_err:.2f}deg <= tol")
            return new_yaw_err

        if prev_yaw_err is not None and new_yaw_err > prev_yaw_err:
            yaw_increase_count += 1
            if yaw_increase_count >= MAX_YAW_INCREASES:
                env_state = DONE_STATE
                print(
                    f"[ALIGN] Overshoot! {yaw_increase_count} consecutive increases. "
                    f"yaw_err={new_yaw_err:.2f}deg"
                )
                return new_yaw_err
        else:
            yaw_increase_count = 0

        prev_yaw_err = new_yaw_err
        return new_yaw_err

    # ── Main REPL ─────────────────────────────────────────────────────────
    while simulation_app.is_running():
        # Auto-continue yaw alignment without prompting
        if env_state == ALIGNING_YAW:
            err = greedy_align_one()
            if env_state == DONE_STATE:
                print(f"[NAV] Navigation complete. Steps={total_steps}")
                _, pos_err, yaw_err = u._is_goal_reached_3act(
                    env_id=0, pos_tol_m=args_cli.pos_tol, yaw_tol_deg=args_cli.yaw_tol_deg
                )
                print(f"[NAV] Final: pos_err={pos_err:.3f}m yaw_err={yaw_err:.2f}deg")
            continue

        cmd = input("[CMD f/l/r/p/n/m/a/s/x/q] > ").strip().lower()

        if cmd == "q":
            break

        if cmd == "f":
            step_action(0)
            continue

        if cmd == "l":
            step_action(2)
            continue

        if cmd == "r":
            step_action(3)
            continue

        if cmd == "x":
            step_action(5)
            continue

        if cmd == "s":
            _save_debug_frame(
                env, save_dir=args_cli.save_dir, frame_idx=frame_idx, action_label="snapshot"
            )
            continue

        # ── Plan with inflation fallback ──────────────────────────────────
        if cmd == "p":
            plan_attempts = 0
            result = try_plan()
            if result and result != ["DONE"]:
                plan_queue = result
                env_state = NAVIGATING
                print(f"[PLAN] Queued {len(plan_queue)} actions")
            elif result == ["DONE"]:
                env_state = DONE_STATE
                print("[NAV] Already at goal!")
            continue

        # ── Next planned action ───────────────────────────────────────────
        if cmd == "n":
            if not plan_queue:
                print("[PLAN] Queue is empty. Run 'p' first.")
                continue
            a = plan_queue.pop(0)
            disp = step_action(a)
            # Collision-revert detection
            if a == 0 and disp is not None and disp < 0.01:
                plan_queue.clear()
                print(f"[REVERT] Forward reverted (disp={disp:.4f}m). Queue cleared for re-plan.")
            # Check if queue drained
            if not plan_queue and env_state == NAVIGATING:
                _, pos_err, yaw_err = u._is_goal_reached_3act(
                    env_id=0, pos_tol_m=args_cli.pos_tol, yaw_tol_deg=args_cli.yaw_tol_deg
                )
                if pos_err <= args_cli.pos_tol and yaw_err <= args_cli.yaw_tol_deg:
                    env_state = DONE_STATE
                    print(f"[NAV] Done! pos_err={pos_err:.3f}m yaw_err={yaw_err:.2f}deg")
                elif pos_err <= args_cli.pos_tol:
                    env_state = ALIGNING_YAW
                    prev_yaw_err = yaw_err
                    yaw_increase_count = 0
                    print(f"[NAV] Position reached, aligning yaw (err={yaw_err:.2f}deg)")
            else:
                print(f"[PLAN] Remaining actions: {len(plan_queue)}")
            continue

        # ── Mass-execute queued actions ───────────────────────────────────
        if cmd == "m":
            if not plan_queue:
                print("[PLAN] Queue is empty. Run 'p' first.")
                continue
            while plan_queue and simulation_app.is_running():
                a = plan_queue.pop(0)
                disp = step_action(a)
                # Collision-revert detection
                if a == 0 and disp is not None and disp < 0.01:
                    plan_queue.clear()
                    print(f"[REVERT] Forward reverted (disp={disp:.4f}m). Queue cleared.")
                    break
                # Early termination check
                reached, pos_err, yaw_err = u._is_goal_reached_3act(
                    env_id=0, pos_tol_m=args_cli.pos_tol, yaw_tol_deg=args_cli.yaw_tol_deg
                )
                if reached:
                    env_state = DONE_STATE
                    print(f"[NAV] Done early! pos_err={pos_err:.3f}m yaw_err={yaw_err:.2f}deg")
                    break
            # Queue drained — transition state
            if not plan_queue and env_state == NAVIGATING:
                _, pos_err, yaw_err = u._is_goal_reached_3act(
                    env_id=0, pos_tol_m=args_cli.pos_tol, yaw_tol_deg=args_cli.yaw_tol_deg
                )
                if pos_err <= args_cli.pos_tol and yaw_err <= args_cli.yaw_tol_deg:
                    env_state = DONE_STATE
                    print(f"[NAV] Done! pos_err={pos_err:.3f}m yaw_err={yaw_err:.2f}deg")
                elif pos_err <= args_cli.pos_tol:
                    env_state = ALIGNING_YAW
                    prev_yaw_err = yaw_err
                    yaw_increase_count = 0
                    print(
                        f"[NAV] Position reached, aligning yaw (err={yaw_err:.2f}deg). "
                        f"Press Enter or 'a' to continue alignment."
                    )
                else:
                    print("[PLAN] Queue drained but position not reached. Re-plan with 'p'.")
            print("[PLAN] Finished replaying queued actions.")
            continue

        # ── Full auto-navigate ────────────────────────────────────────────
        if cmd == "a":
            plan_attempts = 0
            env_state = NAVIGATING
            max_auto_steps = 200
            while (
                env_state != DONE_STATE
                and total_steps < max_auto_steps
                and simulation_app.is_running()
            ):
                if env_state == NAVIGATING:
                    if not plan_queue:
                        result = try_plan()
                        if result == ["DONE"]:
                            env_state = DONE_STATE
                            print("[AUTO] Already at goal!")
                            break
                        if not result:
                            print("[AUTO] Plan failed. Giving up.")
                            env_state = DONE_STATE
                            break
                        plan_queue = result
                    if plan_queue:
                        a = plan_queue.pop(0)
                        disp = step_action(a)
                        if a == 0 and disp is not None and disp < 0.01:
                            plan_queue.clear()
                            print(f"[AUTO-REVERT] Forward reverted (disp={disp:.4f}m). Re-planning.")
                            continue
                        # Check if done after each step
                        _, pos_err, yaw_err = u._is_goal_reached_3act(
                            env_id=0, pos_tol_m=args_cli.pos_tol, yaw_tol_deg=args_cli.yaw_tol_deg
                        )
                        if pos_err <= args_cli.pos_tol and yaw_err <= args_cli.yaw_tol_deg:
                            env_state = DONE_STATE
                            print(f"[AUTO] Done! pos_err={pos_err:.3f}m yaw_err={yaw_err:.2f}deg")
                        elif pos_err <= args_cli.pos_tol:
                            env_state = ALIGNING_YAW
                            prev_yaw_err = yaw_err
                            yaw_increase_count = 0
                            print("[AUTO] Position reached, aligning yaw...")
                elif env_state == ALIGNING_YAW:
                    greedy_align_one()

            if env_state == DONE_STATE:
                _, pos_err, yaw_err = u._is_goal_reached_3act(
                    env_id=0, pos_tol_m=args_cli.pos_tol, yaw_tol_deg=args_cli.yaw_tol_deg
                )
                print(
                    f"[AUTO] Complete. Steps={total_steps} "
                    f"pos_err={pos_err:.3f}m yaw_err={yaw_err:.2f}deg"
                )
            continue

        print("[WARN] Unknown command.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
