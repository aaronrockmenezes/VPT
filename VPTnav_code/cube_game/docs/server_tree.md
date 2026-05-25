# Server Tree

This is the operational layout visible on the HPC server in VS Code.

## Top-Level VPT Workspace

The local repo now mirrors this shape at `/Users/aaronrockmenezes/Desktop/Serre Lab/VPT/`, including a flat `a_star_data_collection_scripts/` directory for planning parity.

```text
/data/arock3/VPT/
├── a_star_data_collection_scripts/
│   ├── __pycache__/
│   ├── audit_astar_valid_starts.py
│   ├── audit_cam_deadzone.py
│   ├── carve_astar_vpt1_firstframe_probe.py
│   ├── compile_a_star_dataset.py
│   ├── compile_tasks.py
│   ├── compute_all_nodes.py
│   ├── count_successful_envs.py
│   └── sanity_check_compiled_a_star.py
├── scripts/
├── v17_v4_logs/
├── VPT_code/
└── VPTnav_code/
    └── cube_game/
```

`a_star_data_collection_scripts/` is a flat operational mirror for frequently run A* utilities. The canonical source copies live under `VPTnav_code/cube_game/scripts/` unless the server workflow explicitly uses the flat mirror.

## Cube Game Job Array

```text
/data/arock3/VPT/VPTnav_code/cube_game/job_array/
├── slurm_logs/
├── a_star/
│   ├── README.md
│   ├── a_star_launcher.py
│   ├── a_star_multi_gpu.sh
│   ├── a_star_worker.sh
│   ├── submit_a_star_array.sh
│   └── submit_a_star.sh
└── normal_vptnav/
    ├── README.md
    ├── generation_worker.sh
    ├── launcher.py
    ├── multi_gpu.sh
    └── submit_generation.sh
```

A* collection uses:

```text
job_array/a_star/submit_a_star_array.sh -> a_star_worker.sh -> a_star_multi_gpu.sh -> a_star_launcher.py -> scripts/A_star_data_collector.py
```

Legacy/non-A* generation uses:

```text
job_array/normal_vptnav/submit_generation.sh -> generation_worker.sh -> multi_gpu.sh -> launcher.py
```

## Cube Game Scripts

```text
/data/arock3/VPT/VPTnav_code/cube_game/scripts/
├── sb3/
├── skrl/
├── A_star_agent.py
├── A_star_automatic_agent.py
├── A_star_data_collector.py
├── compile_results.py
├── launcher_2.py
├── list_envs.py
└── ... A* compile/audit/sanity utilities ...
```

Some A* compile/audit utilities are mirrored into `/data/arock3/VPT/a_star_data_collection_scripts/` for convenience. When changing an A* utility, confirm which copy the server command is using.

## Practical Rule

Before running a command copied from docs, verify the file exists at the path being invoked:

```bash
ls -lh /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/A_star_data_collector.py
ls -lh /users/arock3/data/arock3/VPT/a_star_data_collection_scripts/count_successful_envs.py
```

If both copies exist, prefer the path used by the active job script to avoid version skew.
