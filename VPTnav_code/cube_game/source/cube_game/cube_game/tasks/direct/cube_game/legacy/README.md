# Legacy Cube-Game Task Modules

This folder holds inactive historical task/env/config variants moved out of the
active direct-task package root.

Active task modules stay one directory up:

- normal VPT1: `vpt_env_v18.py`
- depth VPT1: `vpt_env_v18_depth.py`
- A*: `vpt_env_v18_A_star.py`, `vpt_env_v18_A_star_strategy.py`
- camera sweep/optim: `vpt_env_v18_camera.py`, `vpt_env_v18_cam_optim.py`
- VPT2: `vpt2_env_v1.py` through `vpt2_env_v4.py`
- active configs: `vpt_env_cfg_v15_rl.py`, `vpt_env_cfg_v17.py`,
  `vpt2_env_cfg_v1.py`, `vpt2_env_cfg_v2.py`

These files are kept for reference only. Do not register new jobs against
modules in this folder unless they are intentionally promoted back to the active
package root.
