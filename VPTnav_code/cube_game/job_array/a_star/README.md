# A* Job Array

Active A* dataset generation lives here.

Pipeline:

```text
submit_a_star_array.sh -> a_star_worker.sh -> a_star_multi_gpu.sh -> a_star_launcher.py -> scripts/a_star/A_star_data_collector.py
```

Use this directory for `VPT-v18-A-star` rollout collection only. Normal VPTnav
generation lives in `../normal_vptnav/`.

Launch from this directory or through the root wrapper:

```bash
cd /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/job_array/a_star
bash submit_a_star_array.sh

cd /users/arock3/data/arock3/VPT
bash scripts/a_star/submit_v18_a_star.sh
```
