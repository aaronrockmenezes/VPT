# Camera-Move Generation Optimization Report

Date: 2026-05-19

## Context

The current `VPT-v18-camera-move` pipeline is already partially vectorized, but the
measured saved-frame balance (`1282` Yes / `31` No) exposed a deeper throughput problem:
occluded viewpoints are rare, and the env was previously paying for expensive sim/render
checks before discovering that most scenes only produce visible frames.

The recent balanced-save change fixes the output distribution by accepting only envs with
at least one `Yes` and one `No`, then saving equal counts. That makes saved data 50/50, but
it will reduce throughput unless generation finds occlusion-capable scenes earlier.

This report covers how to optimize env generation and checking, especially with vectorized
tensor geometry before sim/image checks.

## Current Pipeline Shape

Primary file:

- `source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move.py`

Key functions:

- `_reset_idx_internal`: full spawn/retry path.
- `initial_spawn_loop`, `move_vpt_objects`, `write_pose_to_sim`: object placement and sim write.
- `geometric_occlusion_check`: vectorized candidate-count precheck around the goal.
- `generate_valid_circle_points`: generates candidate fixed-agent poses.
- `_is_point_valid_batch` / `_check_geometric_validity`: vectorized boundary and distance filters.
- `_check_fov_validity`: slow sim/render loop for candidate agent poses.
- `_build_fixed_sweep_trajectory_batch`: vectorized camera sweep over ready envs.
- `check_batch_object_visibility`: batched semantic-pixel check for agent POV visibility.
- `_camera_goal_visible_from_current_pov`: per-env camera POV red-pixel label check.
- `_check_collisions_vectorized`: GPU tensor 2D SAT collision check against cached obstacle OBBs.

The good news: the batch sweep path is already a strong improvement over the old per-env
loop. It processes every ready env for each sweep angle together, then caches image tensors
for direct saving.

The main weakness: too many scenes reach the expensive sweep/render stage even though their
geometry is unlikely to yield occlusion. Balanced saving now rejects these all-visible scenes
after classification, but we should reject them earlier.

## Bottlenecks

### 1. Render Checks Are Still Used As Early Filters

`generate_valid_circle_points` performs a geometric pass, then `_check_fov_validity`
teleports agents and renders semantic frames to find fixed-agent poses that see both goal
and camera. That is accurate, but expensive.

For camera-move collection, this FOV render check should be the confirmation step, not the
first serious filter.

### 2. Occlusion Is Detected Late

The final label comes from `_camera_goal_visible_from_current_pov`, after camera placement,
sim settle, sensor update, and pixel counting. That is the ground truth label, but a geometric
occlusion predictor can cheaply estimate whether any sweep angles are likely to become `No`.

Right now the env can spend full sweep cost on scenes with zero `No` frames.

### 3. Candidate Generation Still Has Python Per-Env Loops

Several stages build tensor candidates in batch, then loop over envs to sort, queue, or update
status. Some loops are harmless, but loops that wrap sim steps, `.item()`, repeated
`torch.where`, or per-env sensor updates can become expensive at high `NUM_ENVS`.

### 4. Debug Work Runs In Production Path

`debug_plot_analytic_obbs(vpt_state)` is called in `_reset_idx_internal` after agent placement.
If this writes plots or touches matplotlib in production, it should be behind a debug flag.

### 5. The Current Success Criterion Is Output-Balanced, Not Yield-Optimized

Balanced save mode guarantees saved `Yes == No`, but if only 2.4% of generated frames are
occluded, most work is discarded. The next optimization target is not "balance after the
fact"; it is "increase probability that accepted scenes contain occluding sweep angles."

## Recommended Design

Use a three-stage filter:

1. Pure geometric filters on GPU tensors.
2. Cheap analytic visibility/occlusion scoring.
3. Render/pixel validation only for candidates that passed the cheap filters.

The render result remains the source of truth. Geometry should decide where to spend render
budget.

## Stage A: Vectorized Geometry First

For each env, generate candidate sweep camera positions around the goal before rendering:

- Use current fixed sweep angles (`0..180` every `15 deg`) or a denser temporary set.
- Compute camera XY for all envs and all angles as tensors.
- Apply boundary checks with `scene.env_origins` and `center_to_boundary`.
- Apply agent-FOV geometry before camera placement:
  - agent yaw known
  - camera center vector from agent known
  - reject angles outside `half_hfov - buffer`
- Apply 2D OBB collision with `_check_collisions_vectorized`.

This is close to what `_build_fixed_sweep_trajectory_batch` already does. The change is to
run this as an early scoring pass before sim settle and sensor update.

Output per env:

- `num_candidate_angles`
- `candidate_angle_mask`
- `candidate_camera_xy`
- `candidate_camera_quat`

Reject envs with too few candidate angles before rendering.

## Stage B: Analytic Occlusion Precheck

Add a vectorized line-segment vs obstacle-OBB test for camera-to-goal rays.

For each candidate camera position:

- Segment start: camera XY.
- Segment end: goal XY.
- Obstacles: active VPT object OBBs from `obb_corners_cache`.
- Test whether the segment intersects any active obstacle footprint.
- Ignore obstacles that are too close to the camera object or goal by a small margin if needed.

This predicts likely `No` frames without rendering. It will not be perfect because final
labels use semantic pixels, object height, sensor pose, and visual thresholds, but it is
cheap enough to be conservative:

- If predicted occluded count is `0`, reject the scene early.
- If predicted visible count is `0`, reject the scene early.
- If both exist, send only a balanced candidate subset to render validation.

Target precheck:

```text
pred_yes >= CAMERA_MOVE_MIN_PAIRS_PER_ENV
pred_no  >= CAMERA_MOVE_MIN_PAIRS_PER_ENV
```

This directly matches balanced-save mode.

Implementation location:

- Add helper near `_check_collisions_vectorized`:
  - `_segment_intersects_obbs_2d(camera_xy, goal_xy, env_ids) -> Bool[N]`
- Reuse `self.obb_corners_cache[env_ids]`.
- Use active obstacle indices only, not all inactive objects, to avoid false positives.

## Stage C: Render Only Balanced Candidate Sets

Instead of rendering every safe sweep angle, select a small balanced set from the geometric
precheck:

- Choose up to `K` predicted-visible angles and `K` predicted-occluded angles.
- Default `K=1` or `K=2` while debugging yield.
- Preserve angle order for saving.
- Render/pixel-check only selected candidates.

If final pixel labels disagree with the prediction:

- Keep only final `Yes/No` balanced pairs.
- If no balanced pair remains, reject env and retry.

This keeps the render source of truth while minimizing wasted sensor work.

## Stage D: Batch Candidate Validation

The batch sweep currently loops over angles and processes all envs for each angle. That is
reasonable for sensor update cost. Once candidate subsets are smaller and variable-length,
use a flattened candidate table:

```text
candidate_env_ids: [M]
candidate_angle_deg: [M]
candidate_camera_pos: [M, 3]
candidate_camera_quat: [M, 4]
candidate_expected_label: [M]
```

Then process candidates in chunks:

- Place chunk camera objects with `write_root_com_pose_to_sim`.
- Set occlusion sensors for the same env IDs.
- Step/update sensors.
- Run agent POV visibility check in batch.
- Run camera POV red-pixel count in batch, if possible.

This avoids stepping for all 13 angles when only 2-4 balanced candidates are needed.

## Stage E: Batch Camera POV Label Counting

`_camera_goal_visible_from_current_pov` currently handles one env at a time. Add a batched
variant:

```python
def _camera_goal_visible_from_current_pov_batch(self, env_ids):
    sem = self._occlusion_camera.data.output["semantic_segmentation"][env_ids][..., :3]
    ...
    red_counts = red_mask.sum(dim=(1, 2))
    visible = red_counts >= self.goal_pixel_threshold_occlusion
    return visible, red_counts
```

This removes Python iteration for final label counting inside `_build_fixed_sweep_trajectory_batch`.

## Stage F: Early Retry Accounting

Balanced mode means many scenes will be rejected. Track why:

- `no_geometric_candidates`
- `no_predicted_occlusion`
- `no_predicted_visible`
- `render_label_mismatch`
- `agent_view_failed`
- `collision_failed`
- `saved_balanced`

Write a small per-worker JSON summary under:

```text
{BASE_PATH}/logs/generation_stats_node{NODE_ID}_gpu{GPU_ID}.json
```

This is needed because otherwise low yield looks like random slowness.

## High-Impact Implementation Order

### Pass 1: Instrumentation

Add counters for rejection reasons and accepted `Yes/No` pairs. This is the safest first
change because it tells us where time is going.

Expected output per worker:

```text
attempted_scenes
accepted_scenes
rejected_no_predicted_occlusion
rejected_no_final_balanced_pair
avg_candidate_angles_before_render
avg_rendered_angles_per_accepted_env
```

### Pass 2: Batched Camera POV Label Count

Replace per-env red-pixel counting in the sweep loop with a batch function.

Risk: low. It should preserve labels exactly if threshold logic is identical.

### Pass 3: Analytic Segment-vs-OBB Occlusion Precheck

Implement 2D segment intersection against active obstacle OBB footprints. Use it only as a
reject-if-zero-occlusion filter at first.

Risk: medium. Geometry can false-positive/false-negative relative to semantic render. Keep
render as final source of truth.

### Pass 4: Render Candidate Subsets

After precheck is reliable, render only selected predicted `Yes`/`No` candidates instead of
all 13 sweep angles.

Risk: medium-high. This changes dataset coverage over angles. Mitigate by logging angle
histograms and optionally increasing `K`.

### Pass 5: Remove Production Debug Plotting

Gate `debug_plot_analytic_obbs(vpt_state)` behind an environment variable:

```text
CAMERA_MOVE_DEBUG_OBB=1
```

Risk: low.

### Pass 6: Use One Fixed-Agent Viewpoint For Camera-Move

The fixed-agent placement stage only uses `points_2d[0]` after
`generate_valid_circle_points`, then the camera object performs the sweep. Requiring 13
FOV-valid fixed-agent candidate positions was inherited from image-count assumptions and
is unnecessarily strict for camera-move.

Default optimized setting:

```text
CAMERA_MOVE_MIN_AGENT_VIEWPOINTS=1
CAMERA_MOVE_MAX_AGENT_VIEWPOINTS=10
CAMERA_MOVE_MIN_GEOMETRIC_AGENT_CANDIDATES=1
```

Flow:

1. FOV validation collects up to `CAMERA_MOVE_MAX_AGENT_VIEWPOINTS` fixed-agent poses.
2. Envs that already reached that target stop stepping; only remaining envs continue through
   the FOV loop.
3. Each candidate fixed-agent pose gets a geometric 13-angle camera sweep check.
4. Candidate poses are eligible only if they have at least one predicted visible camera
   point and one predicted occluded camera point.
5. Among eligible poses, keep only the candidates tied for the maximum number of valid
   geometric camera sweep points, with predicted balanced pairs and occluded count as
   tie-breakers.

The trajectory-level render validation now renders the normal 15-degree right-half sweep
candidate set and saves a fixed number of frames:

```text
CAMERA_MOVE_TARGET_FRAMES_PER_ENV=12
CAMERA_MOVE_MIN_PAIRS_PER_ENV=1
```

So the agent-placement stage needs one good fixed pose, while the camera-sweep stage saves
12 final frames only when the final semantic labels include at least one `Yes` and one `No`.
It does not force a 50/50 split.

## Expected Impact

If occluded frames remain around 2-3% under save-all behavior, balanced mode without early
occlusion precheck wastes most rendered frames. The analytic precheck should improve
throughput by rejecting all-visible scenes before sweep rendering.

The largest speedups should come from:

1. Rejecting scenes with zero predicted occlusion before sensor work.
2. Rendering only balanced candidate subsets.
3. Batching camera POV red-pixel labels.
4. Disabling production debug work.

## Validation Plan

Run two small jobs on a fresh output root:

1. Baseline balanced mode, current code:
   - `NUM_NODES=1`, `NUM_GPUS=1`, small `NUM_ENVS`.
   - Count saved envs and frames.
   - Save logs for attempts/minute and accepted envs/minute.

2. Optimized precheck mode:
   - Same resources and wall time.
   - Compare accepted envs/minute, frames/minute, and final label balance.

Success criteria:

- `count_saved_envs.py` reports `Yes == No`.
- Accepted envs/minute improves.
- Final camera POV labels still match semantic-pixel source of truth.
- Angle distribution is not collapsed to one tiny region unless intentionally configured.

## Recommendation

Do not loosen the pixel threshold just to get more `No`. Keep semantic render as final
truth. Optimize by predicting occlusion geometrically before rendering, then validate with
pixels.

The practical next patch should be:

1. Add rejection/yield counters.
2. Add batched camera POV red-pixel counting.
3. Add a conservative segment-vs-OBB precheck that rejects scenes with zero predicted
   occlusion.
