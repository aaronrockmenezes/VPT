import torch
import numpy as np

def check_collisions_batched(agents, goal_balls, goal_mats, boundary_limits=None, mat_side=0.6):
    """
    Check for collisions across all environments.
    
    Args:
        agents: Agent rigid objects with position data
        goal_balls: Goal ball rigid objects with position data
        goal_mats: Goal mat assets with position data
        boundary_limits: [x_min, x_max, y_min, y_max] for wall boundaries (relative to each environment's center)
        mat_side: Side length of the square goal mat
    
    Returns:
        collision_mask: Boolean tensor indicating which environments have collisions
        collision_types: List of collision types per environment ("wall", "mat", "goal", None)
    """
    if boundary_limits is None:
        boundary_limits = [-2.0, 2.0, -2.0, 2.0]  # Default boundaries
    
    side = torch.abs(torch.tensor(boundary_limits).view(-1)[0])
    boundary_offset = side - 0.2  # Distance from center to boundary (instead of absolute coordinates)
    
    # Get positions for all environments
    agent_pos = agents.data.root_pos_w  # Shape: (num_envs, 3)
    goal_pos = goal_balls.data.root_pos_w  # Shape: (num_envs, 3)
    mat_pos = goal_mats.data.root_pos_w  # Shape: (num_envs, 3)
    
    num_envs = agent_pos.shape[0]
    device = agent_pos.device
    
    # Object dimensions and collision thresholds
    agent_size = 0.1  # Agent is 0.1x0.1x0.1 cube
    goal_radius = 0.1  # Goal ball radius
    agent_radius = agent_size / 2  # Agent's effective radius for collision
    
    # Goal collision buffer
    goal_buffer = 0.05  # Additional buffer for goal collision
    
    # Initialize results
    collision_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    collision_types = [None] * num_envs
    
    # Check wall collisions relative to each environment's goal position
    # Calculate relative positions of agents to their respective goal positions
    agent_relative_pos = agent_pos - goal_pos  # Shape: (num_envs, 3)
    wall_buffer = 0.05
    
    wall_collision = (
        (agent_relative_pos[:, 0] < -boundary_offset + wall_buffer) |  # Left wall
        (agent_relative_pos[:, 0] > boundary_offset - wall_buffer) |   # Right wall
        (agent_relative_pos[:, 1] < -boundary_offset + wall_buffer) |  # Bottom wall
        (agent_relative_pos[:, 1] > boundary_offset - wall_buffer)     # Top wall
    )
    
    # Check goal mat collisions (within mat boundaries)
    mat_half_side = mat_side / 2
    mat_collision_threshold = 0.15   # Agent radius + small buffer for mat collision
    mat_collision = (
        (torch.abs(agent_pos[:, 0] - mat_pos[:, 0]) < mat_half_side + mat_collision_threshold) &
        (torch.abs(agent_pos[:, 1] - mat_pos[:, 1]) < mat_half_side + mat_collision_threshold)
    )
    
    # Improved goal collision detection (2D only, ignoring Z-axis)
    # Use bounding box collision with buffer
    goal_collision_threshold = agent_radius + goal_radius + goal_buffer  # 0.05 + 0.05 + 0.05 = 0.15
    goal_distance_xy = torch.norm(agent_pos[:, :2] - goal_pos[:, :2], dim=1)
    goal_collision = goal_distance_xy < goal_collision_threshold
    
    # Set collision types in priority order (goal > mat > wall)
    # Goal should have highest priority since it's the success condition
    for env_id in range(num_envs):
        if goal_collision[env_id]:
            collision_mask[env_id] = True
            collision_types[env_id] = "goal"
            # print(f"Goal collision in env ID = {env_id}")
        elif mat_collision[env_id]:
            collision_mask[env_id] = True
            collision_types[env_id] = "mat"
            # pass
        elif wall_collision[env_id]:
            collision_mask[env_id] = False
            collision_types[env_id] = "wall"
            # pass
    
    return collision_mask, collision_types


def check_collisions(entities, mat_side=1.0):
    """Legacy single-environment collision check (kept for compatibility)"""
    agent_ball = entities["agent_ball"]
    goal_ball = entities["goal_ball"]

    agent_pos = agent_ball.data.root_pos_w[0]
    goal_pos = goal_ball.data.root_pos_w[0]

    # Assuming mat is a static object at origin with size mat_side x mat_side
    mat_boundaries = torch.tensor(
        [[-mat_side/2, -mat_side/2, 0.0], 
        [mat_side/2, mat_side/2, 0.0], 
        [mat_side/2, -mat_side/2, 0.0], 
        [-mat_side/2, mat_side/2, 0.0]], 
        device=agent_ball.device
    )

    goal_collision_threshold = 0.18  # Agent half-diagonal (0.05*sqrt(2)) + goal radius (0.1) + small buffer
    mat_collision_threshold = 0.12  # Half agent width (0.05) + half mat height (0.05) + buffer

    goal_distance = torch.norm(agent_pos[:2] - goal_pos[:2])
    mat_distances = torch.norm(agent_pos[:2] - mat_boundaries[:, :2], dim=1)
    mat_distance = mat_distances.min()

    # Check collisions in priority order
    if mat_distance < mat_collision_threshold:
        print("Collision with Mat!")
        return None, "mat"  # Return None for reward (handled in step method)
    elif goal_distance < goal_collision_threshold:
        print("Collision with Goal Ball!")
        return None, "goal"  # Return None for reward (handled in step method)
    else:
        return None, None  # No collision