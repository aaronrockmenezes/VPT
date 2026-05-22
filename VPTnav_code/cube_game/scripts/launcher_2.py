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
    args = parser.parse_args()

    procs = []
    files = []

    # Safe teardown if you hit Ctrl+C
    def cleanup(signum, frame):
        print("\nCaught interrupt. Terminating all child GPU processes...")
        for p in procs:
            p.terminate()
        for f in files:
            f.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"Starting {args.script_path} on {args.num_gpus} GPUs inside Apptainer...")

    for i in range(args.num_gpus):
        # Stagger the launches to prevent Isaac Sim cache deadlocking
        if i > 0:
            print(f"Waiting 15 seconds before launching GPU {i} to prevent cache deadlocks...")
            time.sleep(30)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        env["GPU_ID"] = str(i)

        out_file = os.path.join(args.data_dir, f"data_{i}.txt")
        f = open(out_file, "w")
        files.append(f)

        # sys.executable uses the active Python interpreter native to the IsaacLab container
        cmd = [sys.executable, "-u", args.script_path]
        
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append(p)
        print(f"Launched child process on GPU {i} (PID: {p.pid}) -> data_{i}.txt")

    print("\nAll processes launched! Waiting for completion. Press Ctrl+C to stop all.")
    
    # Block and wait for all child processes to finish naturally
    for p in procs:
        p.wait()

    for f in files:
        f.close()

if __name__ == "__main__":
    main()