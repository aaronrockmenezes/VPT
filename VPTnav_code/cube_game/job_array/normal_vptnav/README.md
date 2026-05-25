# Normal VPTnav Job Array

Active non-A* VPTnav dataset generation lives here.

Pipeline:

```text
submit_generation.sh -> generation_worker.sh -> multi_gpu.sh -> launcher.py -> scripts/keyboard_agent.py
```

Use this directory for normal `VPT-v18`, `VPT-v18-Depth`, and `VPT2-*`
generation. A* rollout collection lives in `../a_star/`. The sibling
`VPTnav_code/cube_game/scripts/` directory is intentionally unchanged and still
holds the agent/data-collector implementation files.

Launch from this directory by editing `submit_generation.sh`, or use the root
wrappers:

```bash
cd /users/arock3/data/arock3/VPT
bash scripts/vptnav/submit_vpt1_v18.sh
bash scripts/vptnav/submit_vpt1_v18_depth.sh
bash scripts/vptnav/submit_vpt2.sh
```
