# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""A* data collector — navigates envs, saves images per --save flag.

Output structure:
  {base_path}/RGB/{Yes|No}/{rollout|rollout_failed}/env_{folder_idx}/step_*.png
  {base_path}/Semantic/{Yes|No}/{rollout|rollout_failed}/env_{folder_idx}/step_*.png
  {base_path}/cam/{Yes|No}/{rollout|rollout_failed}/env_{folder_idx}/{actions.txt,meta.json,final_cam_semantic.png}
  {base_path}/successful_envs.json   # cumulative tracker (success only)

Flags:
  --save pass    only successes saved to rollout/ (failures rmtree'd)
  --save all     successes -> rollout/, failures moved to rollout_failed/
  --img_size N   downsample saved images to NxN (RGB=INTER_AREA, sem=INTER_NEAREST)

Per RL-reset: default A* start is valid_viewpoint_0 from the v18/VPTnav
viewpoint pipeline, requiring camera + goal in frame 0. The legacy random
near-camera spawn remains available only via --start_mode random_near_camera.
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
from concurrent.futures import ThreadPoolExecutor
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
parser.add_argument("--spawn_half_extent",
                    type=float,
                    default=5.0,
                    help="Legacy --start_mode random_near_camera half extent.")
parser.add_argument("--spawn_deadzone",
                    type=float,
                    default=2.0,
                    help="Legacy --start_mode random_near_camera deadzone.")
parser.add_argument(
    "--start_mode",
    choices=["valid_viewpoint", "random_near_camera"],
    default="valid_viewpoint",
    help="A* rollout start. Default uses verified VPTnav-style "
    "valid_viewpoint_0; random_near_camera is debug only.")
parser.add_argument("--start_half_extent",
                    type=float,
                    default=6.0,
                    help="For valid_viewpoint starts, require abs(dx/dy) <= "
                    "this many meters from camera. Default: 6.")
parser.add_argument("--start_deadzone",
                    type=float,
                    default=3.0,
                    help="For valid_viewpoint starts, reject if abs(dx) and "
                    "abs(dy) are both below this. Default: 3.")
parser.add_argument("--cam_no_red_max",
                    type=int,
                    default=0,
                    help="For No categories (occluded/outside_fov), require "
                    "camera POV red pixels <= this value. Default: 0.")
parser.add_argument("--plan_workers",
                    type=int,
                    default=16,
                    help="Thread pool size for parallel A* planning.")
parser.add_argument("--save",
                    choices=["pass", "all"],
                    default="pass",
                    help="'pass': save only successful eps. "
                    "'all': also save failed eps to rollout_failed/.")
parser.add_argument("--img_size",
                    type=int,
                    default=256,
                    help="Resize saved images to NxN. 0 = native res.")
parser.add_argument("--settle_steps",
                    type=int,
                    default=30,
                    help="Extra sim/render steps before saving each rollout "
                    "frame and final cam POV. Helps reduce shimmering "
                    "after movement. Default: 30.")
parser.add_argument("--enforce_split",
                    action="store_true",
                    default=False,
                    help="Enforce per-category 50/25/25 quotas. Default: "
                    "off — save every success, balance at compile time.")
parser.add_argument("--global_target",
                    type=int,
                    default=0,
                    help="If >0, scan compiled JSONs across this base_path "
                    "and dynamically reweight env's visibility pool "
                    "toward under-collected categories. Stops "
                    "wasting compute on already-full categories.")
parser.add_argument("--dynamic_balance_alpha",
                    type=float,
                    default=0.5,
                    help="Adaptive catch-up blend for --global_target. "
                    "0 uses final deficits only; 1 only samples categories "
                    "lagging the current target ratio. Default: 0.5.")
parser.add_argument("--frac_in_view",
                    type=float,
                    default=0.50,
                    help="Generation/category fraction for in_view (Yes).")
parser.add_argument("--frac_occluded",
                    type=float,
                    default=0.25,
                    help="Generation/category fraction for occluded (No).")
parser.add_argument("--frac_outside_fov",
                    type=float,
                    default=0.25,
                    help="Generation/category fraction for outside_fov (No).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import cube_game.tasks  # noqa: F401

import cv2
import gymnasium as gym
import torch

NOOP_ACTION = 4
ACTION_LABELS = {0: "fwd", 1: "back", 2: "left", 3: "right", 4: "noop"}

NAVIGATING = 0
ALIGNING_YAW = 1
DONE_STATE = 2


def _normalize_target_fractions():
    """Return validated category fractions from CLI args."""
    raw = {
        "in_view": max(0.0, float(args_cli.frac_in_view)),
        "occluded": max(0.0, float(args_cli.frac_occluded)),
        "outside_fov": max(0.0, float(args_cli.frac_outside_fov)),
    }
    total = sum(raw.values())
    if total <= 0:
        print("[FATAL] category fractions sum to zero.")
        simulation_app.close()
        sys.exit(2)
    return {k: v / total for k, v in raw.items()}


TARGET_FRACTIONS = _normalize_target_fractions()

# ── helpers ──────────────────────────────────────────────────────────────────


def _compute_yaw_err(u, env_idx):
    return abs(
        math.degrees(
            u._normalize_angle(
                u._get_camera_corrected_yaw(env_idx) -
                u._get_agent_yaw(env_idx))))


def _reason_for(u, i):
    return u.env_visibility_reasons.get(u.slot_folder_indices[i], "unknown")


def _label_for(u, i):
    return u.env_visibility_labels.get(u.slot_folder_indices[i], "unknown")


def _resolve_collection_root(base_path):
    """Return collection root whether passed global root or one GPU output dir."""
    data_dir = os.path.join(base_path, "data")
    if os.path.isdir(data_dir):
        return base_path
    p = os.path.abspath(base_path)
    if os.path.basename(os.path.dirname(p)) == "data":
        return os.path.dirname(os.path.dirname(p))
    return base_path


def _node_id_from_output_dir(name):
    """Extract node id from data_node{node}_gpu* or data_node{node}_compiled."""
    if not name.startswith("data_node"):
        return None
    body = name[len("data_node"):]
    if body.endswith("_compiled"):
        return body[:-len("_compiled")]
    if "_gpu" in body:
        return body.split("_gpu", 1)[0]
    return None


def _compute_category_deficits(base_path, fractions, global_target):
    """Scan compiled/raw JSONs, sum saved per category, compute deficits.

    Dynamic reweighting may be called from a per-GPU raw output directory. This
    resolves back to the collection root and scans `{root}/data`. Compiled dirs
    are authoritative; raw GPU trackers are counted only for nodes that have not
    compiled yet, so running/just-finished jobs can suppress already-filled
    categories before the final per-task compile completes.
    """
    import glob
    root = _resolve_collection_root(base_path)
    data_dir = os.path.join(root, "data")
    saved = {c: 0 for c in fractions}

    compiled_paths = glob.glob(
        os.path.join(data_dir, "data_node*_compiled", "successful_envs.json"))
    compiled_nodes = {
        _node_id_from_output_dir(os.path.basename(os.path.dirname(p)))
        for p in compiled_paths
    }

    paths = list(compiled_paths)
    for p in glob.glob(
            os.path.join(data_dir, "data_node*_gpu*", "successful_envs.json")):
        node_id = _node_id_from_output_dir(os.path.basename(
            os.path.dirname(p)))
        if node_id not in compiled_nodes:
            paths.append(p)

    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
            for c in saved:
                saved[c] += int((d.get("by_reason") or {}).get(c, 0))
        except Exception:
            pass
    targets = {c: int(round(global_target * f)) for c, f in fractions.items()}
    deficits = {c: max(0, targets[c] - saved[c]) for c in fractions}
    return saved, targets, deficits


def _build_dynamic_pool(deficits,
                        min_size=2000,
                        saved=None,
                        fractions=None,
                        alpha=0.5):
    """Build a small shuffled visibility-label pool weighted by deficits.

    Categories with 0 deficit get 0 entries → not generated.
    """
    total_def = sum(deficits.values())
    if total_def <= 0:
        return None
    pool_size = max(1, int(min_size))
    base_weights = {
        cat: (deficit / total_def if deficit > 0 else 0.0)
        for cat, deficit in deficits.items()
    }

    catchup_weights = None
    if saved is not None and fractions is not None:
        total_saved = sum(int(saved.get(cat, 0)) for cat in deficits)
        if total_saved > 0:
            lag = {}
            for cat in deficits:
                if deficits[cat] <= 0:
                    lag[cat] = 0.0
                    continue
                expected_now = total_saved * float(fractions.get(cat, 0.0))
                lag[cat] = max(0.0, expected_now - float(saved.get(cat, 0)))
            lag_total = sum(lag.values())
            if lag_total > 0:
                catchup_weights = {
                    cat: lag[cat] / lag_total
                    for cat in deficits
                }

    alpha = min(1.0, max(0.0, float(alpha)))
    if catchup_weights is None:
        weights = base_weights
    else:
        weights = {
            cat:
            (1.0 - alpha) * base_weights[cat] + alpha * catchup_weights[cat]
            for cat in deficits
        }

    labels = []
    for cat, weight in weights.items():
        if weight <= 0:
            continue
        n = max(1, int(round(weight * pool_size)))
        labels.extend([cat] * n)
    random.shuffle(labels)
    return labels


def _ep_dirs(u, folder_idx, label, subdir="rollout"):
    """Return RGB / Semantic / cam dirs for given folder_idx + subdir."""
    base = f"{u.base_path}/{{}}/{label}/{subdir}/env_{folder_idx}"
    return {
        "rgb": base.format("RGB"),
        "semantic": base.format("Semantic"),
        "cam": base.format("cam")
    }


def _encode_step_frames(u, env_slot, step_n, action_label, img_size):
    """Read agent RGB+semantic, downsample, return list of
    (kind, fname, png_bytes) tuples. No disk I/O.
    """
    rgb = u._rgb_tiled_camera.data.output["rgb"][env_slot]
    sem = u._rgb_tiled_camera.data.output["semantic_segmentation"][env_slot]

    rgb_np = u._resize_rgb(u._to_uint8_rgb(rgb), img_size)
    sem_np = u._resize_sem(u._to_uint8_rgb(sem), img_size)

    fname = f"step_{step_n:05d}_{action_label}.png"
    ok_rgb, rgb_buf = cv2.imencode(".png",
                                   cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    ok_sem, sem_buf = cv2.imencode(".png",
                                   cv2.cvtColor(sem_np, cv2.COLOR_RGB2BGR))
    out = []
    if ok_rgb:
        out.append(("rgb", fname, rgb_buf.tobytes()))
    if ok_sem:
        out.append(("semantic", fname, sem_buf.tobytes()))
    return out


def _settle_before_capture(u, steps):
    """Advance simulation/cameras before image capture.

    Normal non-rollout VPT image collection settles for 30 sim steps after
    teleporting the agent. A* rollout actions are incremental, so the default
    is lower but still gives RTX/lights/semantic buffers time to stabilize.
    """
    for _ in range(max(0, int(steps))):
        u.sim.step()
        u._rgb_tiled_camera.update(u.sim.cfg.dt)
        u._occlusion_camera.update(u.sim.cfg.dt)


def _flush_frame_buffer(u, ep, subdir="rollout"):
    """Write all buffered (kind, fname, bytes) to disk, then clear buffer."""
    if not ep.frame_buffer:
        return
    dirs = _ep_dirs(u, ep.folder_idx, ep.label, subdir)
    os.makedirs(dirs["rgb"], exist_ok=True)
    os.makedirs(dirs["semantic"], exist_ok=True)
    for kind, fname, data in ep.frame_buffer:
        with open(os.path.join(dirs[kind], fname), "wb") as f:
            f.write(data)
    ep.frame_buffer.clear()


def _spawn_agent_near_camera(u, env_idx, rng, half_ext, deadzone):
    """Teleport agent to collision-free XY outside the camera deadzone."""
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
        pos_t = torch.tensor([[wx, wy, z]],
                             device=u.device,
                             dtype=torch.float32)
        if not u._check_collisions_vectorized(ids_t, pos_t, quat)[0]:
            pose_t = torch.cat([pos_t, quat], dim=1)
            u._agent.write_root_com_pose_to_sim(pose_t, ids_t)
            u._agent.write_root_com_velocity_to_sim(
                torch.zeros((1, 6), device=u.device), ids_t)
            return True
    return False


def _refill_label_pool(u,
                       base_path=None,
                       global_target=0,
                       dynamic_balance_alpha=0.5):
    """Top up env's visibility_label_pool. If `global_target > 0`, recompute
    deficits and refill weighted by current shortfall (deficit-only cats);
    else fall back to hardcoded 50/25/25.
    """
    n = max(512, u.num_envs * 4)
    if global_target > 0 and base_path:
        saved, _, deficits = _compute_category_deficits(
            base_path, TARGET_FRACTIONS, global_target)
        new_pool = _build_dynamic_pool(deficits,
                                       min_size=n,
                                       saved=saved,
                                       fractions=TARGET_FRACTIONS,
                                       alpha=dynamic_balance_alpha)
        if new_pool is not None:
            u.visibility_label_pool.extend(new_pool)
            cnt = {c: new_pool.count(c) for c in TARGET_FRACTIONS}
            print(f"[POOL] refilled +{len(new_pool)} (deficit-weighted) "
                  f"saved={saved} deficits={deficits} composition={cnt}")
            return
        print("[POOL] all global category targets full; no labels refilled")
        return
    a = n // 2
    b = n // 4
    c = n - a - b
    new_labels = (["in_view"] * a + ["occluded"] * b + ["outside_fov"] * c)
    random.shuffle(new_labels)
    u.visibility_label_pool.extend(new_labels)
    print(f"[POOL] refilled +{n} labels (default 50/25/25)")


def _advance_slot(u,
                  slot_idx,
                  base_path=None,
                  global_target=0,
                  dynamic_balance_alpha=0.5):
    """Manually replenish slot: bump folder_idx + pull next label from pool.

    Done in lieu of `_replenish_slots` because that one short-circuits when
    `next_env_id >= total_envs_to_sim`. We refill the pool here instead.
    """
    if not u.visibility_label_pool:
        _refill_label_pool(u,
                           base_path=base_path,
                           global_target=global_target,
                           dynamic_balance_alpha=dynamic_balance_alpha)
    if not u.visibility_label_pool:
        return False
    new_env = u.next_env_id
    u.next_env_id += 1
    u.slot_to_env_id[slot_idx] = new_env
    u.slot_folder_indices[slot_idx] = u.next_env_folder_idx + new_env
    u.slot_attempt_counts[slot_idx] = 0
    u.slot_visibility_categories[slot_idx] = u._assign_next_visibility_label(
        u.slot_folder_indices[slot_idx])
    return True


# ── per-env episode state ─────────────────────────────────────────────────────


class Ep:
    __slots__ = ("queue", "plan_attempts", "total_steps", "done", "success",
                 "failed", "fail_reason", "env_state", "align_dir",
                 "align_steps", "best_yaw_err", "align_no_progress",
                 "pre_align_yaw", "pre_step_pos", "actions_taken",
                 "folder_idx", "label", "reason", "frame_buffer", "start_meta")

    def reset(self):
        self.queue = []
        self.plan_attempts = 0
        self.total_steps = 0
        self.done = False
        self.success = False
        self.failed = False
        self.fail_reason = ""
        self.env_state = NAVIGATING
        self.align_dir = None
        self.align_steps = 0
        self.best_yaw_err = None
        self.align_no_progress = 0
        self.pre_align_yaw = None
        self.pre_step_pos = None
        self.actions_taken = []
        self.folder_idx = -1
        self.label = ""
        self.reason = ""
        # Per-step PNG byte buffer used in --save pass mode. List of
        # (kind, fname, png_bytes). Flushed on success; dropped on fail.
        self.frame_buffer = []
        self.start_meta = {}

    def __init__(self):
        self.reset()


# ── planning ──────────────────────────────────────────────────────────────────


def try_plan(u, i, ep, args):
    _, pre_pos, pre_yaw = u._is_goal_reached_3act(env_id=i,
                                                  pos_tol_m=args.pos_tol,
                                                  yaw_tol_deg=args.yaw_tol_deg)
    if pre_pos <= args.pos_tol:
        ep.done = True
        ep.success = True
        return
    if ep.plan_attempts >= args.max_plan_attempts:
        ep.done = True
        ep.failed = True
        ep.fail_reason = "max_plan_attempts"
        return
    if ep.total_steps >= args.max_total_steps:
        ep.done = True
        ep.failed = True
        ep.fail_reason = "max_total_steps_before_plan"
        return

    preplanned = (ep.start_meta or {}).get("preplanned_actions") or []
    if preplanned:
        remaining = args.max_total_steps - ep.total_steps
        ep.queue = [
            int(a) for a in list(preplanned)[:remaining] if a in (0, 2, 3)
        ]
        ep.plan_attempts += 1
        if ep.queue:
            print(
                f"[PLAN-CACHED] env={i} actions={len(ep.queue)} "
                f"candidate={(ep.start_meta or {}).get('astar_candidate_idx')}"
            )
            return
        ep.done = True
        ep.failed = True
        ep.fail_reason = "empty_cached_plan"
        return

    default_infl = float(getattr(u, "planner_inflation_m", 0.12))
    schedule = [None, default_infl * 0.5, default_infl * 0.25, 0.0]
    plan = None
    last_lbl = ""
    for infl in schedule:
        if ep.plan_attempts >= args.max_plan_attempts:
            break
        plan = u.plan_to_camera_actions_3act(env_id=i,
                                             pos_tol_m=args.pos_tol,
                                             yaw_tol_deg=args.yaw_tol_deg,
                                             max_steps=args.max_plan_steps,
                                             inflation_radius=infl)
        ep.plan_attempts += 1
        last_lbl = (f"{infl:.3f}"
                    if infl is not None else f"{default_infl:.3f}(default)")
        if plan.get("success", False):
            print(f"[PLAN-OK] env={i} attempts={ep.plan_attempts} "
                  f"inflation={last_lbl} pre_pos={pre_pos:.3f}")
            break

    if not plan.get("success", False):
        m = plan.get("metrics", {}) or {}
        print(f"[PLAN-FAIL] env={i} attempts={ep.plan_attempts} "
              f"final_inflation={last_lbl} "
              f"expanded={m.get('expanded_nodes','?')} "
              f"pre_pos={pre_pos:.3f}")
        ep.done = True
        ep.failed = True
        ep.fail_reason = f"plan_no_path_{ep.plan_attempts}"
        return

    remaining = args.max_total_steps - ep.total_steps
    ep.queue = [
        int(a) for a in list(plan.get("actions", []))[:remaining]
        if a in (0, 2, 3)
    ]
    if not ep.queue:
        ep.done = True
        ep.failed = True
        ep.fail_reason = f"empty_plan_{ep.plan_attempts}"


# ── tracker JSON ─────────────────────────────────────────────────────────────


class SuccessTracker:
    """Persistent JSON of successful episodes; flushed on every save."""

    def __init__(self, path):
        self.path = path
        self.data = {
            "total": 0,
            "by_reason": {
                "in_view": 0,
                "occluded": 0,
                "outside_fov": 0
            },
            "envs": []
        }
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

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    # Isaac Lab DirectRLEnvCfg exposes a `seed` field. Setting it BEFORE
    # gym.make ensures the env's __init__ calls `configure_seed`, which
    # seeds python random, numpy, torch (CPU+GPU) globally. Without this,
    # the env's heavy use of `random.*` and `np.random` is unseeded and
    # different per-GPU --seed values produce identical trajectories.
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped

    # ── Resume support: if a tracker JSON already exists in this base_path
    # (from a prior chunk), load it and bump next_env_folder_idx so new
    # episodes don't collide with old env_{N} dirs. Tracker must be created
    # BEFORE env.reset() so that _ensure_slot_initialization picks up the
    # offset when assigning the first 96 slot folder_indices.
    tracker = SuccessTracker(os.path.join(u.base_path, "successful_envs.json"))
    if tracker.data["envs"]:
        max_idx = max(e.get("folder_idx", -1) for e in tracker.data["envs"])
        u.next_env_folder_idx = max_idx + 1
        print(f"[RESUME] tracker has {tracker.data['total']} prior envs; "
              f"starting next_env_folder_idx={u.next_env_folder_idx}")

    # ── Dynamic category reweighting based on existing compiled tasks ──
    if args_cli.global_target > 0:
        saved_global, targets_global, deficits = _compute_category_deficits(
            u.base_path, TARGET_FRACTIONS, args_cli.global_target)
        print(f"[DYNAMIC] global saved={saved_global}")
        print(f"[DYNAMIC] global targets={targets_global}")
        print(f"[DYNAMIC] deficits={deficits}")
        new_pool = _build_dynamic_pool(deficits,
                                       min_size=max(512,
                                                    args_cli.num_envs * 4),
                                       saved=saved_global,
                                       fractions=TARGET_FRACTIONS,
                                       alpha=args_cli.dynamic_balance_alpha)
        if new_pool is None:
            print("[DYNAMIC] all categories full; nothing left to collect.")
            env.close()
            return
        # Override env's preallocated pool BEFORE env.reset() so
        # _ensure_slot_initialization pulls from the deficit-weighted pool.
        u.visibility_label_pool = new_pool
        cnt = {c: new_pool.count(c) for c in TARGET_FRACTIONS}
        print(f"[DYNAMIC] override pool size={len(new_pool)}, "
              f"composition={cnt}")

    env.reset(seed=args_cli.seed)
    u._disable_auto_reset = True

    # Bump simulation cap so RL-reset replenishment never blocks
    u.total_envs_to_sim = max(u.total_envs_to_sim,
                              args_cli.target_successes * 10)

    print(f"[INFO] base_path={u.base_path}")
    print(f"[INFO] num_envs={args_cli.num_envs} seed={args_cli.seed} "
          f"target={args_cli.target_successes}")
    print(f"[INFO] save={args_cli.save} img_size={args_cli.img_size} "
          f"plan_workers={args_cli.plan_workers}")
    print(f"[INFO] category_fractions={TARGET_FRACTIONS}")
    print(f"[INFO] start_mode={args_cli.start_mode} "
          f"start_half_extent={args_cli.start_half_extent} "
          f"start_deadzone={args_cli.start_deadzone} "
          f"cam_no_red_max={args_cli.cam_no_red_max} "
          f"settle_steps={args_cli.settle_steps}")

    N = args_cli.num_envs
    quotas = {
        c: int(round(f * args_cli.target_successes))
        for c, f in TARGET_FRACTIONS.items()
    }
    diff = args_cli.target_successes - sum(quotas.values())
    quotas["in_view"] += diff
    saved = tracker.saved_counts()
    for c in TARGET_FRACTIONS:
        saved.setdefault(c, 0)

    eps = [Ep() for _ in range(N)]

    def _quota_met():
        if args_cli.enforce_split:
            return all(saved[c] >= quotas[c] for c in quotas)
        # Total-only: stop when any-category total reaches target.
        return sum(saved.values()) >= args_cli.target_successes

    def _begin_episode(i):
        eps[i].reset()
        eps[i].folder_idx = u.slot_folder_indices[i]
        eps[i].reason = _reason_for(u, i)
        eps[i].label = _label_for(u, i)

    def _capture_initial_frame(i):
        ep = eps[i]
        if args_cli.save == "pass":
            ep.frame_buffer.extend(
                _encode_step_frames(u, i, 0, "start", args_cli.img_size))
        else:
            u.save_rollout_step(i,
                                ep.folder_idx,
                                0,
                                "start",
                                subdir="rollout",
                                img_size=args_cli.img_size,
                                settle_steps=0)

    def _prepare_start(i):
        return bool(_prepare_starts([i]))

    def _prepare_starts(indices):
        indices = [int(i) for i in indices]
        if not indices:
            return []

        if args_cli.start_mode == "random_near_camera":
            ready = []
            for i in indices:
                ep = eps[i]
                ok = _spawn_agent_near_camera(u, i, rng,
                                              args_cli.spawn_half_extent,
                                              args_cli.spawn_deadzone)
                ep.start_meta = {
                    "start_source": "random_near_camera",
                    "start_valid": bool(ok),
                    "start_fail_reason": "" if ok else "spawn_zone_failed",
                    "start_half_extent": float(args_cli.spawn_half_extent),
                    "start_deadzone": float(args_cli.spawn_deadzone),
                    "start_deadzone_metric": "square",
                }
                if ok:
                    ready.append(i)
                else:
                    ep.done = True
                    ep.failed = True
                    ep.fail_reason = "invalid_start:spawn_zone_failed"
                    ep.env_state = DONE_STATE
            if ready:
                _settle_before_capture(u, args_cli.settle_steps)
                for i in ready:
                    _capture_initial_frame(i)
            return ready

        folder_indices = [eps[i].folder_idx for i in indices]
        batch_results = u.prepare_astar_valid_starts(
            indices,
            folder_indices,
            start_half_extent=args_cli.start_half_extent,
            start_deadzone=args_cli.start_deadzone,
            cam_no_red_max=args_cli.cam_no_red_max,
            settle_steps=args_cli.settle_steps,
            pos_tol_m=args_cli.pos_tol,
            yaw_tol_deg=args_cli.yaw_tol_deg,
            max_plan_steps=args_cli.max_plan_steps)

        ready = []
        for i in indices:
            ok, meta = batch_results.get(
                i, (False, {
                    "start_fail_reason": "missing_batch_result"
                }))
            ep = eps[i]
            ep.start_meta = meta
            if ok:
                _capture_initial_frame(i)
                ready.append(i)
            else:
                ep.done = True
                ep.failed = True
                ep.fail_reason = "invalid_start:" + meta.get(
                    "start_fail_reason", "unknown")
                ep.env_state = DONE_STATE
        return ready

    def _do_rl_reset(i):
        if not _advance_slot(
                u,
                i,
                base_path=u.base_path,
                global_target=args_cli.global_target,
                dynamic_balance_alpha=args_cli.dynamic_balance_alpha):
            return
        act = torch.full((N, ), NOOP_ACTION, dtype=torch.long, device=u.device)
        act[i] = 6
        env.step(act)
        _begin_episode(i)
        _prepare_start(i)

    def _commit_success(i, pos_err, yaw_err):
        ep = eps[i]
        # In --save pass mode, per-step PNGs are buffered in RAM; flush
        # them to disk now that this ep is confirmed successful.
        if args_cli.save == "pass":
            _flush_frame_buffer(u, ep, subdir="rollout")
        u.save_rollout_final(i,
                             ep.folder_idx,
                             pos_err,
                             yaw_err,
                             subdir="rollout",
                             img_size=args_cli.img_size,
                             settle_steps=0)

        dirs = _ep_dirs(u, ep.folder_idx, ep.label, "rollout")
        os.makedirs(dirs["cam"], exist_ok=True)
        with open(os.path.join(dirs["cam"], "actions.txt"), "w") as f:
            f.write(" ".join(str(a) for a in ep.actions_taken) + "\n")
        meta = {
            "folder_idx": ep.folder_idx,
            "env_slot": i,
            "seed": args_cli.seed,
            "reason": ep.reason,
            "label": ep.label,
            "success": True,
            "total_steps": ep.total_steps,
            "align_steps": ep.align_steps,
            "plan_attempts": ep.plan_attempts,
            "final_pos_err_m": pos_err,
            "final_yaw_err_deg": yaw_err,
            "img_size": args_cli.img_size,
            "start": ep.start_meta,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        with open(os.path.join(dirs["cam"], "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        tracker.add(meta)
        saved[ep.reason] = saved.get(ep.reason, 0) + 1
        print(f"[SAVED] folder_idx={ep.folder_idx} env={i} reason={ep.reason} "
              f"label={ep.label} | {saved}")

    def _move_to_failed(i, pos_err, yaw_err):
        """--save all path: move discarded ep to rollout_failed/, write meta."""
        ep = eps[i]
        if ep.folder_idx < 0 or ep.label not in ("Yes", "No"):
            return
        src = _ep_dirs(u, ep.folder_idx, ep.label, "rollout")
        dst = _ep_dirs(u, ep.folder_idx, ep.label, "rollout_failed")
        for k in ("rgb", "semantic"):
            if os.path.isdir(src[k]):
                os.makedirs(os.path.dirname(dst[k]), exist_ok=True)
                shutil.rmtree(dst[k], ignore_errors=True)
                shutil.move(src[k], dst[k])
        os.makedirs(dst["cam"], exist_ok=True)
        # Write actions + meta for failed eps too (useful for analysis)
        with open(os.path.join(dst["cam"], "actions.txt"), "w") as f:
            f.write(" ".join(str(a) for a in ep.actions_taken) + "\n")
        meta = {
            "folder_idx": ep.folder_idx,
            "env_slot": i,
            "seed": args_cli.seed,
            "reason": ep.reason,
            "label": ep.label,
            "success": bool(ep.success),
            "failed": bool(ep.failed),
            "fail_reason": ep.fail_reason,
            "total_steps": ep.total_steps,
            "align_steps": ep.align_steps,
            "plan_attempts": ep.plan_attempts,
            "final_pos_err_m": pos_err,
            "final_yaw_err_deg": yaw_err,
            "img_size": args_cli.img_size,
            "start": ep.start_meta,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        with open(os.path.join(dst["cam"], "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _discard(i, pos_err=None, yaw_err=None):
        """Discard or relocate an episode based on --save flag."""
        ep = eps[i]
        if ep.folder_idx < 0 or ep.label not in ("Yes", "No"):
            return
        if args_cli.save == "all":
            _move_to_failed(i, pos_err or 0.0, yaw_err or 0.0)
        else:
            # --save pass: buffered frames never hit disk, just drop them.
            ep.frame_buffer.clear()
            # Defensive cleanup in case anything did slip through.
            dirs = _ep_dirs(u, ep.folder_idx, ep.label, "rollout")
            for d in dirs.values():
                shutil.rmtree(d, ignore_errors=True)

    # Thread pool for parallel A* planning
    plan_pool = ThreadPoolExecutor(max_workers=max(1, args_cli.plan_workers),
                                   thread_name_prefix="a_star_plan")

    def plan_many(indices):
        """Run try_plan in parallel across given env indices."""
        if not indices:
            return
        futures = [
            plan_pool.submit(try_plan, u, i, eps[i], args_cli) for i in indices
        ]
        for fut in futures:
            fut.result()  # surface exceptions

    # Initial setup
    for i in range(N):
        _begin_episode(i)
    initial_ready = _prepare_starts(range(N))
    plan_many(initial_ready)

    sim_steps = 0
    pending_reset = set()  # envs that finished episode, awaiting bulk reset

    while not _quota_met():
        actions = [NOOP_ACTION] * N
        active_this_step = [False] * N

        # ── pre-step ──────────────────────────────────────────────────────
        for i in range(N):
            ep = eps[i]
            if ep.done:
                continue
            if ep.total_steps >= args_cli.max_total_steps:
                ep.done = True
                ep.failed = True
                ep.fail_reason = "max_total_steps"
                ep.env_state = DONE_STATE
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
                agent_yaw = u._get_agent_yaw(i)
                delta = u._normalize_angle(target_yaw - agent_yaw)
                delta_deg = math.degrees(abs(delta))

                if abs(delta) < u.heading_bin_rad / 2.0:
                    pos_err_now = float(
                        torch.norm(
                            u._agent.data.root_pos_w[i, :2] -
                            u._camera_obj.data.root_pos_w[i, :2]).item())
                    ep.done = True
                    ep.success = (pos_err_now <= args_cli.pos_tol
                                  and delta_deg <= args_cli.yaw_tol_deg)
                    ep.env_state = DONE_STATE
                    print(f"[ALIGN-DONE] env={i} yaw_err={delta_deg:.1f}°")
                    continue

                actions[i] = 2 if delta > 0 else 3
                active_this_step[i] = True
                if ep.align_dir is None:
                    pe = float(
                        torch.norm(
                            u._agent.data.root_pos_w[i, :2] -
                            u._camera_obj.data.root_pos_w[i, :2]).item())
                    print(f"[ALIGN-START] env={i} pos_err={pe:.3f}m "
                          f"yaw_err={delta_deg:.1f}°")
                ep.align_dir = actions[i]
                ep.pre_align_yaw = u._get_agent_yaw(i)

        action_tensor = torch.tensor(actions,
                                     dtype=torch.long,
                                     device=u.device)
        env.step(action_tensor)
        sim_steps += 1

        did_capture_settle = False
        if any(active_this_step):
            _settle_before_capture(u, args_cli.settle_steps)
            sim_steps += max(0, int(args_cli.settle_steps))
            did_capture_settle = True

        # ── post-step: per-frame save + revert + alignment ────────────────
        for i in range(N):
            ep = eps[i]

            if active_this_step[i]:
                ep.total_steps += 1
                ep.actions_taken.append(actions[i])
                action_label = ACTION_LABELS.get(actions[i], f"a{actions[i]}")
                try:
                    if args_cli.save == "pass":
                        # Buffer encoded PNG bytes in RAM. Flush on
                        # success, drop on fail. Avoids writing PNGs
                        # for failed eps.
                        ep.frame_buffer.extend(
                            _encode_step_frames(u, i, ep.total_steps,
                                                action_label,
                                                args_cli.img_size))
                    else:
                        # --save all: write directly so failed eps can
                        # be moved to rollout_failed/ at finalization.
                        u.save_rollout_step(i,
                                            ep.folder_idx,
                                            ep.total_steps,
                                            action_label,
                                            subdir="rollout",
                                            img_size=args_cli.img_size,
                                            settle_steps=0)
                except Exception as e:
                    print(f"[WARN] save_rollout_step env={i} failed: {e}")

            # Forward-revert detection
            if (ep.env_state == NAVIGATING and actions[i] == 0
                    and ep.pre_step_pos is not None):
                disp = float(
                    torch.norm(u._agent.data.root_pos_w[i, :2].cpu() -
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
            pos_err_now = float(
                torch.norm(u._agent.data.root_pos_w[i, :2] -
                           u._camera_obj.data.root_pos_w[i, :2]).item())

            if ep.pre_align_yaw is not None:
                yd = abs(
                    u._normalize_angle(u._get_agent_yaw(i) - ep.pre_align_yaw))
                ep.align_no_progress = (ep.align_no_progress + 1 if
                                        yd < u.heading_bin_rad * 0.25 else 0)
            ep.pre_align_yaw = None

            if ep.align_no_progress >= 4:
                ep.done = True
                ep.env_state = DONE_STATE
                print(f"[ALIGN-STUCK] env={i} yaw_err={yaw_err_now:.1f}°")
                continue

            new_delta = u._normalize_angle(
                u._get_camera_corrected_yaw(i) - u._get_agent_yaw(i))
            new_dir = 2 if new_delta > 0 else 3

            if new_dir != actions[i]:
                ep.done = True
                ep.success = (pos_err_now <= args_cli.pos_tol
                              and yaw_err_now <= args_cli.yaw_tol_deg)
                ep.env_state = DONE_STATE
                print(f"[ALIGN-STOP] env={i} overshoot "
                      f"yaw_err={yaw_err_now:.1f}° steps={ep.align_steps}")
                continue

            if ep.best_yaw_err is None or yaw_err_now < ep.best_yaw_err:
                ep.best_yaw_err = yaw_err_now

            if ep.align_steps >= args_cli.max_align_steps:
                ep.done = True
                ep.success = (pos_err_now <= args_cli.pos_tol
                              and yaw_err_now <= args_cli.yaw_tol_deg)
                ep.env_state = DONE_STATE
                print(f"[ALIGN-MAX] env={i} yaw_err={yaw_err_now:.1f}°")

        # ── post-step: queue-drained transitions ──────────────────────────
        to_replan_drained = []
        for i in range(N):
            ep = eps[i]
            if ep.done or ep.env_state != NAVIGATING or ep.queue:
                continue

            _, pos_err, yaw_err = u._is_goal_reached_3act(
                env_id=i,
                pos_tol_m=args_cli.pos_tol,
                yaw_tol_deg=args_cli.yaw_tol_deg)

            if pos_err <= args_cli.pos_tol:
                if yaw_err <= args_cli.yaw_tol_deg:
                    ep.done = True
                    ep.success = True
                    ep.env_state = DONE_STATE
                    print(f"[ALIGN-SKIP] env={i} yaw_err={yaw_err:.1f}°")
                    continue
                ep.env_state = ALIGNING_YAW
                ep.align_dir = None
                ep.align_steps = 0
                ep.best_yaw_err = yaw_err
                ep.align_no_progress = 0
                print(f"[ALIGN-START] env={i} pos_err={pos_err:.3f}m "
                      f"yaw_err={yaw_err:.1f}°")
                continue

            if ep.total_steps >= args_cli.max_total_steps:
                ep.done = True
                ep.failed = True
                ep.fail_reason = "max_total_steps_after_exec"
                ep.env_state = DONE_STATE
                continue
            if ep.plan_attempts >= args_cli.max_plan_attempts:
                ep.done = True
                ep.failed = True
                ep.fail_reason = "max_plan_attempts_after_exec"
                ep.env_state = DONE_STATE
                continue

            to_replan_drained.append(i)

        plan_many(to_replan_drained)
        for i in to_replan_drained:
            ep = eps[i]
            if not ep.queue and not ep.done:
                ep.done = True
                ep.failed = True
                ep.fail_reason = "empty_replan"
                ep.env_state = DONE_STATE

        # ── episode finalization (commit/discard envs that just finished) ─
        # Note: actual RL reset is deferred until ALL envs are done, to
        # batch them into one synchronous reset round (see below).
        finalizing_now = [
            i for i in range(N) if eps[i].done and i not in pending_reset
        ]
        if finalizing_now and not did_capture_settle:
            _settle_before_capture(u, args_cli.settle_steps)
            sim_steps += max(0, int(args_cli.settle_steps))

        for i in range(N):
            ep = eps[i]
            if not ep.done or i in pending_reset:
                continue

            _, pos_err, yaw_err = u._is_goal_reached_3act(
                env_id=i,
                pos_tol_m=args_cli.pos_tol,
                yaw_tol_deg=args_cli.yaw_tol_deg)

            if args_cli.enforce_split:
                within_quota = (saved.get(ep.reason, 0) < quotas.get(
                    ep.reason, 0))
            else:
                within_quota = True  # save all successes regardless of cat

            if ep.success and within_quota:
                _commit_success(i, pos_err, yaw_err)
                print(f"[DONE] env={i} reason={ep.reason} SUCCESS "
                      f"steps={ep.total_steps} align={ep.align_steps}")
            else:
                tag = "OVERFLOW" if ep.success else "FAIL"
                print(f"[DONE] env={i} reason={ep.reason} {tag} "
                      f"steps={ep.total_steps} failed={ep.failed} "
                      f"fail_reason={ep.fail_reason}")
                _discard(i, pos_err, yaw_err)

            pending_reset.add(i)

        if _quota_met():
            break

        # Synchronous bulk reset: only when ALL envs have finished their
        # episodes. Done envs sit idle (NOOP) until the slowest finishes,
        # then we reset all N at once. Avoids per-tick reset stalls and
        # the global USD churn from staggered resets.
        if len(pending_reset) == N:
            to_reset = sorted(pending_reset)
            print(f"[SYNC-RESET] all {N} envs done -> bulk reset")
            if args_cli.global_target > 0:
                # Force fresh deficit-only labels each bulk reset. This lets
                # long-running array jobs stop producing a category shortly
                # after earlier jobs compile/fill it.
                u.visibility_label_pool.clear()
                _refill_label_pool(
                    u,
                    base_path=u.base_path,
                    global_target=args_cli.global_target,
                    dynamic_balance_alpha=args_cli.dynamic_balance_alpha)
            no_more_labels = False
            for i in to_reset:
                if not _advance_slot(
                        u,
                        i,
                        base_path=u.base_path,
                        global_target=args_cli.global_target,
                        dynamic_balance_alpha=args_cli.dynamic_balance_alpha):
                    no_more_labels = True
                    break
            if no_more_labels:
                print("[DYNAMIC] no remaining category deficits; stopping.")
                break
            act = torch.full((N, ),
                             NOOP_ACTION,
                             dtype=torch.long,
                             device=u.device)
            for i in to_reset:
                act[i] = 6
            env.step(act)
            sim_steps += 1
            for i in to_reset:
                _begin_episode(i)
            reset_ready = _prepare_starts(to_reset)
            plan_many(reset_ready)
            pending_reset.clear()

    plan_pool.shutdown(wait=True)
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
