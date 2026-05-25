"""Spawn NUM_GPUS A* data collector subprocesses for ONE array task.

Per-GPU:
  - CUDA_VISIBLE_DEVICES = i
  - seed = SEED_BASE + TASK_ID*1000 + i (unique across (task, gpu))
  - target_successes = ceil(TASK_TARGET / NUM_GPUS)
  - 30s stagger to avoid Isaac kit cache deadlocks.

Each task writes outputs under the env-derived
{BASE_PATH}/data/data_node{JOB_ID}_{TASK_ID}_gpu{GPU}/ path, so no two
submissions share a folder unless JOB_ID is intentionally reused.
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--job_id",
                   type=str,
                   default="0",
                   help="SLURM_ARRAY_JOB_ID; namespaces output dirs.")
    p.add_argument("--num_gpus", type=int, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--task_target", type=int, required=True)
    p.add_argument("--num_envs", type=int, required=True)
    p.add_argument("--plan_workers", type=int, required=True)
    p.add_argument("--max_total_steps", type=int, required=True)
    p.add_argument("--img_size", type=int, required=True)
    p.add_argument("--settle_steps", type=int, default=30)
    p.add_argument("--start_mode",
                   choices=["valid_viewpoint", "random_near_camera"],
                   default="valid_viewpoint")
    p.add_argument("--start_half_extent", type=float, default=6.0)
    p.add_argument("--start_deadzone", type=float, default=3.0)
    p.add_argument("--cam_no_red_max", type=int, default=0)
    p.add_argument("--save", type=str, required=True, choices=["pass", "all"])
    p.add_argument("--seed_base", type=int, required=True)
    p.add_argument("--global_target",
                   type=int,
                   default=0,
                   help="Total goal across all array tasks; enables dynamic "
                   "category reweighting in the collector.")
    p.add_argument("--frac_in_view", type=float, default=0.50)
    p.add_argument("--frac_occluded", type=float, default=0.25)
    p.add_argument("--frac_outside_fov", type=float, default=0.25)
    p.add_argument("--dynamic_balance_alpha",
                   type=float,
                   default=0.5,
                   help="Blend from remaining-deficit targeting (0.0) to "
                   "catch-up targeting for underrepresented classes (1.0).")
    args = p.parse_args()

    per_gpu = math.ceil(args.task_target / args.num_gpus)
    script = "/mnt/VPT/VPTnav_code/cube_game/scripts/A_star_data_collector.py"

    print(f"[LAUNCHER] task={args.task_id} gpus={args.num_gpus} "
          f"per_gpu_target={per_gpu} (task_target {args.task_target})")

    procs = []
    files = []

    def cleanup(signum, frame):
        print("\n[LAUNCHER] caught signal, terminating children...")
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
        for f in files:
            try:
                f.close()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    for gpu_id in range(args.num_gpus):
        if gpu_id > 0:
            print(f"[LAUNCHER] sleeping 30s before GPU {gpu_id} "
                  f"(kit cache stagger)")
            time.sleep(30)

        # Fold job_id into seed so different SLURM submissions don't
        # reuse seeds for the same (task, gpu) coords.
        try:
            job_int = int(args.job_id)
        except ValueError:
            job_int = abs(hash(args.job_id)) % (10**6)
        seed = (args.seed_base + (job_int % 10**4) * 100000 +
                args.task_id * 1000 + gpu_id)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["GPU_ID"] = str(gpu_id)
        # Override NODE_ID so the env's base_path points to a unique
        # subdir per (job, task, gpu). The env builds:
        #   {BASE_PATH}/data/data_node{NODE_ID}_gpu{GPU_ID}/
        # NODE_ID = {SLURM_ARRAY_JOB_ID}_{task_id} → no collisions
        # across multiple sbatch submissions to the same BASE_PATH.
        env["NODE_ID"] = f"{args.job_id}_{args.task_id}"

        out_path = os.path.join(args.data_dir,
                                f"data_node{args.task_id}_gpu{gpu_id}.txt")
        f = open(out_path, "w", buffering=1)
        files.append(f)

        cmd = [
            sys.executable,
            "-u",
            script,
            "--task",
            args.task,
            "--num_envs",
            str(args.num_envs),
            "--target_successes",
            str(per_gpu),
            "--plan_workers",
            str(args.plan_workers),
            "--max_total_steps",
            str(args.max_total_steps),
            "--img_size",
            str(args.img_size),
            "--settle_steps",
            str(args.settle_steps),
            "--start_mode",
            args.start_mode,
            "--start_half_extent",
            str(args.start_half_extent),
            "--start_deadzone",
            str(args.start_deadzone),
            "--cam_no_red_max",
            str(args.cam_no_red_max),
            "--save",
            args.save,
            "--seed",
            str(seed),
            "--global_target",
            str(args.global_target),
            "--dynamic_balance_alpha",
            str(args.dynamic_balance_alpha),
            "--frac_in_view",
            str(args.frac_in_view),
            "--frac_occluded",
            str(args.frac_occluded),
            "--frac_outside_fov",
            str(args.frac_outside_fov),
        ]
        print(f"[LAUNCHER] task={args.task_id} gpu={gpu_id} "
              f"seed={seed} -> {out_path}")
        proc = subprocess.Popen(cmd,
                                env=env,
                                stdout=f,
                                stderr=subprocess.STDOUT)
        procs.append(proc)

    print("[LAUNCHER] all spawned. Waiting for completion.")
    rcs = [proc.wait() for proc in procs]
    for f in files:
        f.close()
    print(f"[LAUNCHER] task={args.task_id} exit codes: {rcs}")
    sys.exit(0 if all(rc == 0 for rc in rcs) else 1)


if __name__ == "__main__":
    main()
