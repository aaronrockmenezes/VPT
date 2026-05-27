# VPT Project TODO

This is the repo-local backlog for project cleanup, documentation, and
operations. Keep private/local project knowledge here or under
`VPTnav_code/cube_game/docs/`; do not use Obsidian as the source of truth unless
explicitly requested.

## High Priority

- [ ] Migrate Oscar server tree to the canonical GitHub-backed repo.
      See `VPTnav_code/cube_game/docs/server_sync_todo.md`.

- [ ] Full cleanup and documentation pass for VPTnav/VPT1/VPT2 workflows.
      Goal: make a private, local repo wiki plus FAQ that a new agent or Aaron
      can use without reconstructing history from chats.

- [ ] LP/FT thesis-reporting leftovers.
      Current VPT1 v18 LP/FT results look promising, but before treating them
      as final thesis evidence: add a train/val/test FT mode, select FT
      checkpoints by validation accuracy instead of test accuracy, rerun the
      clean FT sweep or a focused model subset, document stalled/timeout LP
      models, and fold the ImageNet-correlation plot into the results notes.

## Documentation Cleanup Scope

- [x] Separate and clearly name the active pipelines:
      normal VPTnav/VPT1 generation, VPT2 generation, A* rollout generation,
      camera-move generation, linear probes, and fine-tuning. Active SLURM
      generation scripts are split under
      `VPTnav_code/cube_game/job_array/normal_vptnav/` and
      `VPTnav_code/cube_game/job_array/a_star/`; root wrappers are split under
      `scripts/vptnav/` and `scripts/a_star/`.

- [ ] Document normal VPTnav v18 end-to-end:
      `job_array/normal_vptnav/submit_generation.sh` ->
      `generation_worker.sh` -> `multi_gpu.sh` -> `keyboard_agent.py` ->
      `VPT-v18` -> `scripts/utils/build_vpt1_dataset.py` -> LP/FT.

- [ ] Document VPT2 end-to-end:
      registered task IDs, expected agent script, output layout, compile/carve
      step, label map, LP/FT commands, and known gotchas.

- [ ] Document A* separately so it is not confused with normal v18.
      Include when to use `job_array/a_star/submit_a_star_array.sh`,
      `compile_a_star_dataset.py`, `make_vpt_probe_split.py`, and world-model
      handoff docs.

- [ ] Audit old/stale docs and rewrite or archive misleading sections:
      especially camera-move sync notes that predate the GitHub canonical repo,
      old A* defaults, server paths, and duplicated task names.

- [ ] Create a private local FAQ covering:
      which task to launch for VPT1/VPT2/A*/camera-move, where outputs go, how
      to count data, how to carve 512/1024 datasets, when to use H100 vs
      Blackwell, how to run LP/FT, how to recover from partial SLURM runs, and
      what files should never enter git.

- [ ] Add a command cookbook with copy-paste commands for:
      smoke generation, production generation, counting, dataset build,
      validation, LP, FT, result compilation, and server migration.

- [ ] Add a file map that explains ownership and status of major scripts:
      active, legacy, experimental, stale, or do-not-touch.

- [ ] Reconcile `AGENTS.md`, `VPTnav_code/cube_game/agents.md`,
      `VPTnav_code/cube_game/MEMORY.md`, and `VPTnav_code/cube_game/docs/wiki.md`
      so they agree on the canonical workflows.

## Cleanup Scope

- [ ] Identify generated artifacts currently near source code and ensure they
      are ignored or moved outside git.

- [ ] Decide what to do with legacy duplicated env files and copy-suffixed
      scripts: keep with labels, archive under a clear legacy folder, or remove
      after verification.

- [ ] Verify all runnable server scripts use the canonical root path and do not
      depend on nested repo state.

- [ ] Build a short verification checklist for every dataset before LP/FT:
      count, balance, file structure, image counts, labels, and sample visual QC.
