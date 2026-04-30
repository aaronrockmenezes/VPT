import os
import subprocess
import sys
import signal
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--script_path', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    node_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")

    procs = []
    files = []

    def cleanup(signum, frame):
        print("\nCaught interrupt. Terminating all child GPU processes...")
        for p in procs:
            p.terminate()
        for f in files:
            f.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"Starting {args.script_path} on {args.num_gpus} GPUs | Node {node_id} | Task {args.task}")

    for i in range(args.num_gpus):
        if i > 0:
            print(f"Waiting 30 seconds before launching GPU {i} to prevent cache deadlocks...")
            time.sleep(30)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        env["GPU_ID"] = str(i)
        env["NODE_ID"] = node_id

        out_file = os.path.join(args.data_dir, f"data_node{node_id}_gpu{i}.txt")
        f = open(out_file, "w")
        files.append(f)

        cmd = [sys.executable, "-u", args.script_path, "--task", args.task]
        
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append(p)
        print(f"Launched Node {node_id} GPU {i} (PID: {p.pid}) | Task: {args.task} -> data_node{node_id}_gpu{i}.txt")

    print("\nAll processes launched! Waiting for completion. Press Ctrl+C to stop all.")
    
    for p in procs:
        p.wait()

    for f in files:
        f.close()

if __name__ == "__main__":
    main()