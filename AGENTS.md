# VPT Agent Entry Point

Read this first in new Codex chats.

This repository is the working project context. Work on actual VPT project
files in this directory unless the user explicitly says otherwise.

Do not edit the user's Obsidian/PERMANENT_MEMORY vault or `.obsidian`
content unless the user explicitly asks.

## Durable Project State

Keep durable notes inside this VPT repo. The root project log is:

- `docs/codex_log.md`

Each meaningful log entry should include:

- date
- task
- files changed
- summary
- open questions
- next actions

## Canonical Project Docs

Project docs currently live under:

- `VPTnav_code/cube_game/agents.md`
- `VPTnav_code/cube_game/MEMORY.md`
- `VPTnav_code/cube_game/docs/wiki.md`
- `VPTnav_code/cube_game/docs/world_model_handoff.md`
- `docs/project_todo.md`
- `docs/codex_log.md`

Do not assume cross-chat memory. Treat the markdown files above as the durable project state.

## Current Operational Note

As of 2026-05-22, this root repository is the canonical git source for the VPT
server snapshot. The server tarball `~/Downloads/VPT.tar.gz` was imported as
the base tree with runtime artifacts excluded: nested `.git` directories,
`.simg`/`.sif` binaries, logs, caches, W&B/Hydra outputs, checkpoints,
`v17_v4_logs/`, `v18_logs/`, and duplicate `VPTnav_code/cube_game copy/`.
Do not reintroduce nested repo state; track source files from this root repo.

The active A* array submit script is:

- `VPTnav_code/cube_game/job_array/submit_a_star_array.sh`

It currently derives SLURM array max concurrency from:

```text
MAX_TOTAL_CPUS=140
MAX_TOTAL_GPUS=70
```

With `CPUS_PER_TASK=6` and `NUM_GPUS=4`, this caps the array at 17 concurrent
tasks/nodes via `--array=0-${ARRAY_HI}%${MAX_PARALLEL_NODES}`.

## Cleanup / Documentation TODO

The repo needs a full cleanup and documentation pass, tracked in:

- `docs/project_todo.md`

That pass should produce a private local wiki and FAQ covering normal VPTnav
VPT1 generation, VPT2 generation, A* rollouts, camera-move generation, dataset
count/compile/validation, linear probing, fine-tuning, server migration, and
which generated artifacts must stay out of git.
