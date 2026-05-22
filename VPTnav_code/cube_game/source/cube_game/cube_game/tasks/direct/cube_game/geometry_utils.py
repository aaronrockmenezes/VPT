import torch
from typing import List
import math


def generate_valid_circle_points(
        self,
        env_ids: torch.Tensor,
        angle_step: float = 2.0) -> List[torch.Tensor]:
    """Generate valid viewpoint positions in parallel for all environments."""
    device = self.device
    num_envs = len(env_ids)

    MIN_REQUIRED_POINTS = self.images_per_env

    # Generate all candidate angles
    num_angles = int(360.0 / angle_step)
    angles = torch.linspace(0, 2 * math.pi, num_angles, device=device)

    # Get radii for all environments
    fov_deg = 30.0
    fov_rad = math.radians(fov_deg)
    camera_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
    goal_pos = self._goal.data.root_pos_w[env_ids, :2]
    d = torch.norm(camera_pos - goal_pos, dim=1)

    half_fov = torch.tensor(fov_rad / 2, device=self.device)
    radii = (d / 2) / torch.tan(half_fov)
    radii = radii * 1.3
    radii = radii.unsqueeze(1)

    # Generate all points for all envs
    angles_expanded = angles.unsqueeze(0).expand(num_envs, -1)
    all_x = self._goal.data.root_pos_w[env_ids, 0].unsqueeze(
        1) + radii * torch.cos(angles_expanded)
    all_y = self._goal.data.root_pos_w[env_ids, 1].unsqueeze(
        1) + radii * torch.sin(angles_expanded)

    total_points = num_envs * num_angles
    all_points_batch = torch.stack([all_x, all_y],
                                   dim=2).reshape(total_points, 2)
    env_ids_batch = env_ids.unsqueeze(1).expand(
        -1, num_angles).reshape(total_points)

    # Step 1: Geometric validation
    geometric_valid = _is_point_valid_batch(points=all_points_batch,
                                                 env_ids=env_ids_batch,
                                                 check_agent_fov=False)

    geometric_valid_per_env = geometric_valid.reshape(num_envs, num_angles)

    if self.verbose >= 2:
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()
            # print(
            #     f"  Env {env_id_item}: {geometric_valid_per_env[i].sum().item()} geometric candidates"
            # )

    # Step 2: Vectorized displacement filtering across all environments
    displacement_filtered_points = []
    displacement_filtered_env_ids = []
    displacement_filtered_indices = []

    MIN_CANDIDATES_FOR_FOV = 40  # Require at least this many candidates before FOV check

    for i, env_id in enumerate(env_ids):
        env_id_item = env_id.item()
        valid_mask = geometric_valid_per_env[i]

        if not valid_mask.any():
            if self.verbose >= 2:
                print(f"  Env {env_id_item}: No geometric candidates")
            continue

        num_geometric = valid_mask.sum().item()

        # Early check: need enough geometric candidates
        if num_geometric < MIN_CANDIDATES_FOR_FOV:
            if self.verbose >= 1:
                print(
                    f"  Env {env_id_item}: ❌ Only {num_geometric} geometric candidates, need {MIN_CANDIDATES_FOR_FOV}. Skipping."
                )
            continue

        valid_points = all_points_batch[i * num_angles:(i + 1) *
                                        num_angles][valid_mask]

        # Sort by angle for consistent ordering
        goal_pos_2d = self._goal.data.root_pos_w[env_id, :2]
        relative_pos = valid_points - goal_pos_2d
        angles_rad = torch.atan2(relative_pos[:, 1], relative_pos[:, 0])
        sorted_indices = torch.argsort(angles_rad)
        sorted_points = valid_points[sorted_indices]

        # Vectorized greedy displacement filtering
        num_points = len(sorted_points)
        if num_points == 0:
            continue

        # Compute pairwise distances for all points at once
        # Shape: (num_points, num_points)
        diff = sorted_points.unsqueeze(0) - sorted_points.unsqueeze(1)
        pairwise_distances = torch.norm(diff, dim=2)

        # Greedy selection using vectorized operations
        selected_mask = torch.zeros(num_points,
                                    dtype=torch.bool,
                                    device=device)
        selected_mask[0] = True  # Always select first point

        for idx in range(1, num_points):
            # Check if this point is far enough from all selected points
            distances_to_selected = pairwise_distances[idx, selected_mask]
            if torch.all(distances_to_selected >= self.min_viewpoint_distance):
                selected_mask[idx] = True

        filtered_candidates = sorted_points[selected_mask]

        num_before = len(sorted_points)
        num_after = len(filtered_candidates)

        # Check: need enough candidates after displacement filtering
        if num_after < MIN_CANDIDATES_FOR_FOV:
            if self.verbose >= 1:
                rejection_rate = (
                    1 - num_after / num_before) * 100 if num_before > 0 else 0
                print(
                    f"  Env {env_id_item}: ❌ Only {num_after} candidates after displacement ({rejection_rate:.1f}% rejected), need {MIN_CANDIDATES_FOR_FOV}. Skipping FOV check."
                )
            continue

        if self.verbose >= 2:
            rejection_rate = (
                1 - num_after / num_before) * 100 if num_before > 0 else 0
            print(
                f"  Env {env_id_item}: Displacement filter: {num_after}/{num_before} kept ({rejection_rate:.1f}% rejected)"
            )

        # Add these pre-filtered candidates for FOV checking
        displacement_filtered_points.append(filtered_candidates)
        displacement_filtered_env_ids.extend([env_id.item()] *
                                             len(filtered_candidates))
        displacement_filtered_indices.extend([i] * len(filtered_candidates))

    if len(displacement_filtered_points) == 0:
        if self.verbose >= 1:
            print(
                f"  ❌ No candidates passed displacement filter for any environment"
            )
        return [torch.zeros((0, 2), device=device) for _ in range(num_envs)]

    # Concatenate all displacement-filtered candidates
    all_candidates = torch.cat(displacement_filtered_points, dim=0)
    all_candidates_env_ids = torch.tensor(displacement_filtered_env_ids,
                                          dtype=torch.long,
                                          device=device)
    all_candidates_indices = torch.tensor(displacement_filtered_indices,
                                          dtype=torch.long,
                                          device=device)

    if self.verbose >= 2:
        total_geometric = geometric_valid.sum().item()
        total_after_displacement = len(all_candidates)
        saved_compute = ((total_geometric - total_after_displacement) /
                         total_geometric * 100) if total_geometric > 0 else 0
        print(
            f"  💡 FOV candidates: {total_after_displacement}/{total_geometric} ({saved_compute:.1f}% compute saved)"
        )

    # Store original agent state
    original_agent_pos = self._agent.data.root_pos_w[env_ids].clone()
    original_agent_quat = self._agent.data.root_quat_w[env_ids].clone()

    # Step 3: FOV check ONLY on displacement-filtered candidates
    fov_valid_mask = _is_point_valid_batch(
        points=all_candidates,
        env_ids=all_candidates_env_ids,
        check_agent_fov=True,
        min_required_points=
        MIN_REQUIRED_POINTS  # Pass min required to enable early stopping
    )

    # Restore original agent positions
    self._agent.write_root_pose_to_sim(torch.cat(
        [original_agent_pos, original_agent_quat], dim=-1),
                                       env_ids=env_ids)

    # Step 4: Collect final valid points per environment
    all_valid_points = []

    for i, env_id in enumerate(env_ids):
        env_id_item = env_id.item()

        # Get FOV-valid points for this env
        env_mask = all_candidates_indices == i
        env_fov_valid = fov_valid_mask[env_mask]

        num_candidates = env_mask.sum().item()
        num_fov_valid = env_fov_valid.sum().item()

        if not env_fov_valid.any():
            all_valid_points.append(torch.zeros((0, 2), device=device))
            if self.verbose >= 1:
                print(
                    f"  Env {env_id_item}: ❌ 0/{num_candidates} passed FOV check"
                )
            continue

        valid_points_tensor = all_candidates[env_mask][env_fov_valid]

        fov_rejection_rate = (1 - num_fov_valid / num_candidates
                              ) * 100 if num_candidates > 0 else 0

        if len(valid_points_tensor) >= MIN_REQUIRED_POINTS:
            all_valid_points.append(valid_points_tensor)
            if self.verbose >= 2:
                print(
                    f"  Env {env_id_item}: ✅ {len(valid_points_tensor)}/{num_candidates} passed FOV ({fov_rejection_rate:.1f}% rejected)"
                )
        else:
            all_valid_points.append(torch.zeros((0, 2), device=device))
            if self.verbose >= 1:
                print(
                    f"  Env {env_id_item}: ❌ Only {len(valid_points_tensor)}/{MIN_REQUIRED_POINTS} points ({fov_rejection_rate:.1f}% FOV rejection)"
                )

    return all_valid_points


def _is_point_valid_batch(points: torch.Tensor,
                          env_ids: torch.Tensor,
                          min_obstacle_distance: float = 0.3,
                          min_camera_obstacle_distance: float = 0.4,
                          min_camera_target_distance: float = 1.0,
                          min_target_obstacle_distance: float = None,
                          check_agent_fov: bool = False,
                          min_required_points: int = None) -> torch.Tensor:
    """Pipeline: Geometric Checks (Fast) -> FOV Checks (Slow/Simulated)."""
    if min_required_points is None:
        min_required_points = self.images_per_env

    if min_target_obstacle_distance is None:
        min_target_obstacle_distance = self.goal_radius + 0.01

    # 1. Fast Geometric Checks
    valid_mask = _check_geometric_validity(points, env_ids,
                                                min_obstacle_distance,
                                                min_camera_obstacle_distance,
                                                min_camera_target_distance,
                                                min_target_obstacle_distance)

    if not check_agent_fov or not valid_mask.any():
        return valid_mask

    # 2. Slow FOV Checks (Physics Simulation)
    valid_mask = _check_fov_validity(points, env_ids, valid_mask,
                                          min_required_points)

    return valid_mask


def _check_geometric_validity(self, points, env_ids, min_obs_dist,
                              min_cam_obs_dist, min_cam_target_dist,
                              min_target_obs_dist):
    """Validates boundaries, obstacle proximity, and camera clearance."""
    device = points.device
    valid_mask = torch.ones(points.shape[0], dtype=torch.bool, device=device)
    
    # Boundary Check
    env_origins = self.scene.env_origins[env_ids, :2]
    in_bounds = torch.all((points >= env_origins - self.center_to_boundary) &
                          (points <= env_origins + self.center_to_boundary),
                          dim=1)
    valid_mask &= in_bounds

    if not valid_mask.any(): return valid_mask

    # Active Obstacle Positions
    active_obs_pos = self._get_active_obstacle_positions(env_ids)

    # Point (Goal) -> Obstacle Check
    # Calculates distance from every candidate point to every obstacle
    dist_pt_obs = torch.norm(points.unsqueeze(1) - active_obs_pos, dim=2)

    # Enforce min_target_obs_dist here
    valid_mask &= (dist_pt_obs.min(dim=1)[0] >= min_target_obs_dist)

    # Camera Checks
    cam_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
    valid_mask &= (torch.norm(points - cam_pos, dim=1) >= min_cam_target_dist)

    dist_cam_obs = torch.norm(cam_pos.unsqueeze(1) - active_obs_pos, dim=2)
    valid_mask &= (dist_cam_obs.min(dim=1)[0] >= min_cam_obs_dist)

    return valid_mask


def _check_fov_validity(self, points, env_ids, valid_mask, min_req_points):
    """Simulates view to check visibility. Handles sampling and logging."""
    device = points.device
    points_to_check = torch.where(valid_mask)[0]

    if points_to_check.numel() == 0: return valid_mask

    # Save State
    saved_pos = self._agent.data.root_pos_w[env_ids].clone()
    saved_quat = self._agent.data.root_quat_w[env_ids].clone()

    # Sample & queue points
    env_queues, env_status = _sample_fov_candidates(points,
                                                         env_ids,
                                                         points_to_check,
                                                         max_s=120)

    fov_valid_global = torch.zeros_like(valid_mask)
    max_steps = max(len(q['points']) for q in env_queues.values())

    # Simulation Loop
    for step in range(max_steps):
        if self.verbose >= 2 and step % 10 == 0:
            print(f"    🔄 FOV progress: {step+1}/{max_steps} points")

        # Batch construction
        step_ids, step_pts, step_idxs = [], [], []

        for eid, data in env_queues.items():
            # Skip satisfied envs
            if env_status[eid]['count'] >= min_req_points: continue

            if step < len(data['points']):
                step_ids.append(torch.tensor(eid, device=device))
                step_pts.append(data['points'][step])
                step_idxs.append(data['indices'][step])

        if not step_ids:
            if self.verbose >= 1 and all(s['count'] >= min_req_points
                                         for s in env_status.values()):
                print(
                    f"    ✅ All environments have {min_req_points}+ valid points. Early stop."
                )
            break

        # Vectorize
        b_envs = torch.stack(step_ids)
        b_pts = torch.stack(step_pts)
        b_idxs = torch.stack(step_idxs)

        # Teleport -> Step -> Check
        _teleport_and_step(b_envs, b_pts)
        g_vis, c_vis = self.check_batch_object_visibility(b_envs)
        is_vis = g_vis & c_vis

        # Update results
        fov_valid_global[b_idxs] = is_vis

        # Update counts & Log
        for i, eid in enumerate(b_envs.tolist()):
            if is_vis[i]:
                env_status[eid]['count'] += 1
                if env_status[eid][
                        'count'] == min_req_points:  # Just hit threshold
                    if self.verbose >= 1:
                        print(
                            f"    🎯 Env {eid}: Reached {min_req_points} valid points."
                        )

    # Restore State
    restore = torch.cat([saved_pos, saved_quat], dim=1)
    self._agent.write_root_com_pose_to_sim(restore, env_ids)

    return valid_mask & fov_valid_global


def _sample_fov_candidates(self, points, env_ids, indices, max_s):
    """Groups by env, downsamples, and logs stats."""
    unique_envs = torch.unique(env_ids[indices])
    queues, status = {}, {}

    for eid in unique_envs:
        eid_item = eid.item()
        env_indices = indices[env_ids[indices] == eid]
        total = len(env_indices)

        if total > max_s:
            perm = torch.randperm(total, device=points.device)[:max_s]
            env_indices = env_indices[perm]
            if self.verbose >= 2:
                print(
                    f"  🎲 Env {eid_item}: Sampled {max_s}/{total} candidates.")
        else:
            if self.verbose >= 2:
                print(f"  🎲 Env {eid_item}: Using all {total} candidates.")

        queues[eid_item] = {
            'points': points[env_indices],
            'indices': env_indices
        }
        status[eid_item] = {'count': 0}

    return queues, status


def _teleport_and_step(self, env_ids, points):
    """Teleports agents, orients to midpoints, and steps physics."""
    cam_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
    goal_pos = self._goal.data.root_pos_w[env_ids, :2]

    dirs = ((cam_pos + goal_pos) / 2.0) - points
    yaws = torch.atan2(dirs[:, 1], dirs[:, 0])

    # Pose construction
    pos = torch.zeros((len(env_ids), 3), device=points.device)
    pos[:, :2] = points
    pos[:, 2] = self._agent.data.default_root_state[env_ids, 2]

    quat = torch.zeros((len(env_ids), 4), device=points.device)
    quat[:, 0] = torch.cos(yaws / 2)
    quat[:, 3] = torch.sin(yaws / 2)

    self._agent.write_root_com_pose_to_sim(torch.cat([pos, quat], dim=1),
                                           env_ids)

    # Step Physics
    self.sim.step()
    self._tiled_camera.update(self.sim.cfg.dt)
    self._agent.update(self.sim.cfg.dt)
    self._camera_obj.update(self.sim.cfg.dt)
    self._goal.update(self.sim.cfg.dt)
    self._vpt_objects.update(self.sim.cfg.dt)
