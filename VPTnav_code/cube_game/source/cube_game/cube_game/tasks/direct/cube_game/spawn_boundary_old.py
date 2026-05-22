import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from typing import List
import numpy as np
import random
import os
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

boundary_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2))

@configclass 
class BoundaryWallsCfg:
    """Configuration for boundary walls compatible with InteractiveScene."""
    
    def __init__(self, 
                 boundary_corners: List[List[float]] = [[-2.0, -2.0], [2.0, 2.0]],
                 color: sim_utils.PreviewSurfaceCfg = None,
                 wall_thickness: float = 0.1,
                 wall_height: float = 1.0,
                 prim_prefix = None):
        """
        Initialize boundary walls configuration.
        
        Parameters:
        -----------
        boundary_corners : List[List[float]]
            Diagonal corners of the square boundary [[x_min, y_min], [x_max, y_max]].
        color : sim_utils.PreviewSurfaceCfg
            Visual material configuration for the walls.
        wall_thickness : float
            Thickness of the boundary walls.
        wall_height : float
            Height of the boundary walls.
        """
        if color is None:
            color = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2))
            
        self.boundary_corners = boundary_corners
        self.color = color
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.prim_prefix = prim_prefix if prim_prefix is not None else "{ENV_REFEX_NS}/"
        
        # Calculate wall configurations
        self._setup_walls()
    
    def _setup_walls(self):
        """Setup wall configurations for InteractiveScene."""
        corner1, corner2 = self.boundary_corners
        x_min, y_min = corner1
        x_max, y_max = corner2
        
        # Calculate dimensions
        width = x_max - x_min
        height = y_max - y_min
        
        # Add extra width to north/south walls to cover corners
        corner_overlap = self.wall_thickness  # Overlap by wall thickness to eliminate gaps
        
        # Define wall configurations
        self.wall_configs = {
            # Bottom wall (south) - extended to cover corners
            "bottom_wall": {
                "pos": [(x_min + x_max) / 2, y_min - self.wall_thickness / 2, self.wall_height / 2],
                "size": [width + (2 * corner_overlap), self.wall_thickness, self.wall_height]
            },
            # Top wall (north) - extended to cover corners
            "top_wall": {
                "pos": [(x_min + x_max) / 2, y_max + self.wall_thickness / 2, self.wall_height / 2],
                "size": [width + (2 * corner_overlap), self.wall_thickness, self.wall_height]
            },
            # Left wall (west) - normal size
            "left_wall": {
                "pos": [x_min - self.wall_thickness / 2, (y_min + y_max) / 2, self.wall_height / 2],
                "size": [self.wall_thickness, height, self.wall_height]
            },
            # Right wall (east) - normal size
            "right_wall": {
                "pos": [x_max + self.wall_thickness / 2, (y_min + y_max) / 2, self.wall_height / 2],
                "size": [self.wall_thickness, height, self.wall_height]
            }
        }
    
    def get_wall_rigid_object_cfg(self, wall_name: str) -> RigidObjectCfg:
        """
        Get RigidObjectCfg for a specific wall.
        
        Parameters:
        -----------
        wall_name : str
            Name of the wall (bottom_wall, top_wall, left_wall, right_wall).
            
        Returns:
        --------
        RigidObjectCfg
            Configuration for the wall rigid object.
        """
        if wall_name not in self.wall_configs:
            raise ValueError(f"Wall '{wall_name}' not found. Available walls: {list(self.wall_configs.keys())}")
        
        wall_config = self.wall_configs[wall_name]
        
        return RigidObjectCfg(
            prim_path=f"{self.prim_prefix}/{wall_name}",
            spawn=sim_utils.CuboidCfg(
                size=tuple(wall_config["size"]),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.0,  # Zero mass makes it kinematic/static
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    # kinematic_enabled=True,  # Make it kinematic (immovable)
                    max_linear_velocity=0.0,
                    max_angular_velocity=0.0,
                    solver_position_iteration_count=8,  # Better collision solving
                    solver_velocity_iteration_count=1
                ),
                visual_material=self.color,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=wall_config["pos"],
                rot=(0.0, 0.0, 0.0, 1.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0)
            )
        )
    
    def get_all_wall_configs(self) -> dict:
        """
        Get all wall configurations as RigidObjectCfg.
        
        Returns:
        --------
        dict
            Dictionary of wall names to RigidObjectCfg.
        """
        return {
            wall_name: self.get_wall_rigid_object_cfg(wall_name) 
            for wall_name in self.wall_configs.keys()
        }

def create_boundary_walls_cfg(boundary_corners: List[List[float]] = [[-2.0, -2.0], [2.0, 2.0]],
                             color: sim_utils.PreviewSurfaceCfg = None,
                             wall_thickness: float = 0.1,
                             wall_height: float = 1.0,
                             prim_prefix=None) -> BoundaryWallsCfg:
    """
    Create boundary walls configuration for InteractiveScene.
    
    Parameters:
    -----------
    boundary_corners : List[List[float]]
        Diagonal corners of the square boundary [[x_min, y_min], [x_max, y_max]].
    color : sim_utils.PreviewSurfaceCfg
        Visual material configuration for the walls.
    wall_thickness : float
        Thickness of the boundary walls.
    wall_height : float
        Height of the boundary walls.
        
    Returns:
    --------
    BoundaryWallsCfg
        Configuration object for boundary walls.
    """
    return BoundaryWallsCfg(boundary_corners, color, wall_thickness, wall_height, prim_prefix)

def create_boundary_walls(boundary_corners: List[List[float]] = [[-2.0, -2.0], [2.0, 2.0]],
                         color: sim_utils.PreviewSurfaceCfg = None,
                         wall_thickness: float = 0.1,
                         wall_height: float = 1.0,
                         prim_prefix=None) -> dict:
    """
    Create boundary walls as RigidObjectCfg entities.
    
    Parameters:
    -----------
    boundary_corners : List[List[float]]
        Diagonal corners of the square boundary [[x_min, y_min], [x_max, y_max]].
    color : sim_utils.PreviewSurfaceCfg
        Visual material configuration for the walls.
    wall_thickness : float
        Thickness of the boundary walls.
    wall_height : float
        Height of the boundary walls.
        
    Returns:
    --------
    dict
        Dictionary of wall names to RigidObjectCfg entities.
    """
    if color is None:
        color = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2))
    
    corner1, corner2 = boundary_corners
    x_min, y_min = corner1
    x_max, y_max = corner2
    
    # Calculate dimensions
    width = x_max - x_min
    height = y_max - y_min
    corner_overlap = wall_thickness
    
    # Define wall configurations
    wall_configs = {
        "bottom_wall": {
            "pos": [(x_min + x_max) / 2, y_min - wall_thickness / 2, wall_height / 2],
            "size": [width + (2 * corner_overlap), wall_thickness, wall_height]
        },
        "top_wall": {
            "pos": [(x_min + x_max) / 2, y_max + wall_thickness / 2, wall_height / 2],
            "size": [width + (2 * corner_overlap), wall_thickness, wall_height]
        },
        "left_wall": {
            "pos": [x_min - wall_thickness / 2, (y_min + y_max) / 2, wall_height / 2],
            "size": [wall_thickness, height, wall_height]
        },
        "right_wall": {
            "pos": [x_max + wall_thickness / 2, (y_min + y_max) / 2, wall_height / 2],
            "size": [wall_thickness, height, wall_height]
        }
    }
    
    # Create RigidObjectCfg for each wall
    wall_entities = {}
    for wall_name, wall_config in wall_configs.items():
        wall_entities[wall_name] = RigidObjectCfg(
            prim_path=f"{prim_prefix}/{wall_name}",
            spawn=sim_utils.CuboidCfg(
                size=tuple(wall_config["size"]),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    max_linear_velocity=0.0,
                    max_angular_velocity=0.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1
                ),
                visual_material=color,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=wall_config["pos"],
                rot=(0.0, 0.0, 0.0, 1.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0)
            )
        )
    
    return wall_entities

def get_boundary_cfg(boundary_corners=[[-2.0, -2.0], [2.0, 2.0]], boundary_height=3.0, prim_prefix="/World/envs/env_.*"):
    boundary_cfg = create_boundary_walls_cfg(
            boundary_corners=boundary_corners,
            color=boundary_material,
            wall_thickness=0.2,
            wall_height=boundary_height,
            prim_prefix=prim_prefix
        )
    
    wall_configs = boundary_cfg.get_all_wall_configs()
    bottom_wall, top_wall, left_wall, right_wall = wall_configs.values()
    return bottom_wall, top_wall, left_wall, right_wall

def get_color():
    """
    Generate a random pastel color that is not similar to red, green, blue, or pink.
    Returns a sim_utils.PreviewSurfaceCfg with the chosen color.
    """
    # Define forbidden colors and a threshold for "similarity"
    forbidden_colors = [
        np.array([1.0, 0.0, 0.0]),   # red
        np.array([0.0, 1.0, 0.0]),   # green
        np.array([0.0, 0.0, 1.0]),   # blue
        np.array([0.8, 0.0, 0.0]),   # pinkish-red
        np.array([1.0, 0.75, 0.8]),  # pink
        np.array([1.0, 0.2, 0.6]),   # pink
        np.array([0.9, 0.1, 0.5]),   # pink
        np.array([0.95, 0.3, 0.6]),  # pink
        np.array([0.0, 0.0, 0.0])    # black
    ]
    threshold = 0.2  # Euclidean distance threshold for "similarity"

    valid = False
    while not valid:
        # Generate a pastel color by mixing with white, but limit the mix to avoid being too white
        base = np.array([random.uniform(0.0, 0.9) for _ in range(3)])
        # Check similarity to forbidden colors
        too_close = any(np.linalg.norm(base - fc) < threshold for fc in forbidden_colors)
        # Avoid colors that are too close to pure red/green/blue/pink
        if not too_close:
            valid = True
    return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(base))

def get_shape(fix_shape=None):
    # shapes_list = ["capsule", "cone", "cuboid", "cylinder", "sphere"]
    shapes_list = ["cone", "cuboid", "cylinder", "sphere"]
    # 50% big, 50% small
    cuboid_size = (np.random.uniform(0.75, 1.5, (3,)).tolist()) if random.random() < 0.3 else (np.random.uniform(0.25, 0.75, (3,)).tolist())

    cone_height = np.random.uniform(0.5, 1.5) if random.random() < 0.3 else np.random.uniform(0.2, 0.5)
    cylinder_height = np.random.uniform(0.5, 1.5) if random.random() < 0.3 else np.random.uniform(0.2, 0.5)
    
    cone_radius = np.random.uniform(0.25, 0.75) if random.random() < 0.3 else np.random.uniform(0.1, 0.25)
    cylinder_radius = np.random.uniform(0.25, 0.75) if random.random() < 0.3 else np.random.uniform(0.1, 0.25)
    sphere_radius = np.random.uniform(0.25, 0.75) if random.random() < 0.3 else np.random.uniform(0.1, 0.25)
    
    if fix_shape is not None and fix_shape in shapes_list:
        chosen_shape = fix_shape
    else:
        chosen_shape = random.choice(shapes_list)
    if chosen_shape == "capsule":
        shape = sim_utils.CapsuleCfg(
            radius=0.5,
            height=0.5,
            axis=random.choice(['X', 'Y', 'Z']),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "cone":
        shape = sim_utils.ConeCfg(
            radius=cone_radius,
            height=cone_height,
            axis=random.choice(['X', 'Y', 'Z']),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "cuboid":
        shape = sim_utils.CuboidCfg(
            size=cuboid_size,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "cylinder":
        shape = sim_utils.CylinderCfg(
            radius=cylinder_radius,
            height=cylinder_height,
            axis=random.choice(['X', 'Y', 'Z']),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "sphere":
        shape = sim_utils.SphereCfg(
            radius=sphere_radius,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    return shape

def get_shape_randomize(fix_shape=None):
    # shapes_list = ["capsule", "cone", "cuboid", "cylinder", "sphere"]
    shapes_list = ["cone", "cuboid", "cylinder", "cross", "L", "t"]
    # shapes_list = ["cone", "cuboid", "cylinder", "sphere"]
    # 50% big, 50% small
    cuboid_size = (np.random.uniform(0.1, 2.0, (3,)).tolist())

    cone_height = np.random.uniform(0.1, 2.0)
    cylinder_height = np.random.uniform(0.1, 2.0)
    
    cone_radius = np.random.uniform(0.2, 1.0)
    cylinder_radius = np.random.uniform(0.2, 1.0)
    sphere_radius = np.random.uniform(0.2, 1.0)

    cross_scaling_factors = (1/0.06, 1/0.06, 1/0.020182)    # X, Y and Z scaling factors to be used to get object to (1,1,1)m
    l_scaling_factors = (1/0.021433, 1/0.036474, 1/0.018880)    # X, Y and Z scaling factors to be used to get object to (1,1,1)m
    t_scaling_factors = (1/0.020121, 1/0.034027, 1/0.018880)    # X, Y and Z scaling factors to be used to get object to (1,1,1)m

    usd_cross_scale_xy = np.random.uniform(0.1, 2.0) * cross_scaling_factors[0]
    usd_cross_scale_z = np.random.uniform(0.1, 2.0)  * cross_scaling_factors[2]
    usd_cross_scale = (usd_cross_scale_xy, usd_cross_scale_xy, usd_cross_scale_z)

    usd_l_scale_xy = np.random.uniform(0.1, 2.0)
    usd_l_scale_z = np.random.uniform(0.1, 2.0)  * l_scaling_factors[2]
    usd_l_scale = (usd_l_scale_xy * l_scaling_factors[0], usd_l_scale_xy * l_scaling_factors[1], usd_l_scale_z)

    usd_t_scale_xy = np.random.uniform(0.1, 2.0)
    usd_t_scale_z = np.random.uniform(0.1, 2.0)  * t_scaling_factors[2]
    usd_t_scale = (usd_t_scale_xy * t_scaling_factors[0], usd_t_scale_xy * t_scaling_factors[1], usd_t_scale_z)
    
    if fix_shape is not None and fix_shape in shapes_list:
        chosen_shape = fix_shape
    else:
        chosen_shape = random.choice(shapes_list)
    if chosen_shape == "cone":
        shape = sim_utils.ConeCfg(
            radius=cone_radius,
            height=cone_height,
            # axis=random.choice(['Z']),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_linear_velocity=1.0, max_angular_velocity=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "cuboid":
        shape = sim_utils.CuboidCfg(
            size=cuboid_size,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_linear_velocity=1.0, max_angular_velocity=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "cylinder":
        shape = sim_utils.CylinderCfg(
            radius=cylinder_radius,
            height=cylinder_height,
            # axis=random.choice(['X', 'Y', 'Z']),  # Default is 'Z'
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_linear_velocity=1.0, max_angular_velocity=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "sphere":
        shape = sim_utils.SphereCfg(
            radius=sphere_radius,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_linear_velocity=1.0, max_angular_velocity=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "cross":
        shape = sim_utils.UsdFileCfg(
            usd_path="/home/arock3/cube_game/assets/Cross/Kit1_Cross.usd",
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_linear_velocity=1.0, max_angular_velocity=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            scale=usd_cross_scale,
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "L":
        shape = sim_utils.UsdFileCfg(
            usd_path="/home/arock3/cube_game/assets/L/Kit1_Character_L.usd",
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_linear_velocity=1.0, max_angular_velocity=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            scale=usd_l_scale,
            semantic_tags=[("class", "obstacles")],)
    elif chosen_shape == "t":
        shape = sim_utils.UsdFileCfg(
            usd_path="/home/arock3/cube_game/assets/T/Kit1_Character_T.usd",
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_linear_velocity=1.0, max_angular_velocity=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=get_color(),
            scale=usd_t_scale,
            semantic_tags=[("class", "obstacles")],)
    return shape

def spawn_random_objects(num_objects, boundary_corners, prim_prefix):
    """
    Spawn a given number of random objects inside the boundary.
    Returns a dict of RigidObjectCfgs keyed by object name.
    """
    corner1, corner2 = boundary_corners
    x_min, y_min = corner1
    x_max, y_max = corner2
    objects = {}
    object_metadata = {}
    for i in range(num_objects):
        x = random.uniform(x_min + 0.2, x_max - 0.2)
        y = random.uniform(y_min + 0.2, y_max - 0.2)
        z = 0.0  # spawn at ground level
        obj_name = f"random_obj_{i}"
        obj_cfg = RigidObjectCfg(
            prim_path=f"{prim_prefix}/{obj_name}",
            spawn=get_shape(fix_shape="cuboid"),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x, y, z),
                rot=(0.0, 0.0, 0.0, 1.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0)
            )
        )
        objects[obj_name] = obj_cfg
        object_metadata[obj_name] = {
            "name": obj_name,
            "type": "random",
            "x": x,
            "y": y,
            "z": z,
            "shape": obj_cfg.spawn.__class__.__name__,
            "size": obj_cfg.spawn.size,
            "color": obj_cfg.spawn.visual_material.diffuse_color
        }
    return objects, object_metadata

def spawn_random_objects_random(num_objects, boundary_corners, prim_prefix, num_options=100):
    """
    Spawn a given number of random objects inside the boundary.
    Returns a dict of RigidObjectCfgs keyed by object name.
    """
    corner1, corner2 = boundary_corners
    x_min, y_min = corner1
    x_max, y_max = corner2
    objects = {}
    object_metadata = {}
    for i in range(num_objects):
        x = random.uniform(x_min + 0.2, x_max - 0.2)
        y = random.uniform(y_min + 0.2, y_max - 0.2)
        z = 0.0  # spawn at ground level
        obj_name = f"random_obj_{i}"
        obj_cfg = RigidObjectCfg(
            prim_path=f"{prim_prefix}/{obj_name}",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[get_shape_randomize(fix_shape="cuboid") for _ in range(num_options)],
                random_choice=True,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                semantic_tags=[("class", "obstacles")]
            ),  
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x, y, z),
                rot=(0.0, 0.0, 0.0, 1.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0)
            )
        )
        objects[obj_name] = obj_cfg
        object_metadata[obj_name] = {
            "name": obj_name,
            "type": "random",
            "x": x,
            "y": y,
            "z": z,
            # "shape": obj_cfg.spawn.__class__.__name__,
            # "size": obj_cfg.spawn.size,
            # "color": obj_cfg.spawn.visual_material.diffuse_color
        }
    return objects, object_metadata


def get_rigid_obj(prim_path, fix_shape=None):
    shape = get_shape_randomize(fix_shape=fix_shape)
    
    if fix_shape in ["cross", "L", "t"]:
        # Generate random angle (0 to 360 degrees)
        random_angle = np.random.uniform(0, 2 * np.pi)
    
        if fix_shape == "cross":
            # Generate random yaw (0 to 360 degrees)
            random_yaw = np.random.uniform(0, 2 * np.pi)
            
            # Step 1: Stand up (90° pitch around Y)
            q_pitch = np.array([0.0, 0.707, 0.0, 0.707])
            
            # Step 2: Roll 45° around Z
            q_roll = euler_to_quaternion(0, 0, np.pi/4)
            
            # Step 3: Random yaw around X
            q_yaw = euler_to_quaternion(random_yaw, 0, 0)
            
            # Combine: pitch -> roll -> yaw
            q_temp = quaternion_multiply(q_pitch, q_roll)
            rot_array = quaternion_multiply(q_temp, q_yaw)
            rot = tuple(rot_array)
        elif fix_shape == "L":
            # Generate random yaw (0 to 360 degrees)
            random_yaw = np.random.uniform(0, 2 * np.pi)
            
            # Step 1: Stand up (90° around Y)
            q_base = np.array([0.0, 0.707, 0.0, 0.707])
            
            # Step 2: Random yaw around X (additional rotation)
            q_yaw = euler_to_quaternion(random_yaw, 0, 0)
            
            # Combine: stand up then yaw
            rot_array = quaternion_multiply(q_base, q_yaw)
            rot = tuple(rot_array)
        elif fix_shape == "t":
            # Generate random yaw (0 to 360 degrees)
            random_yaw = np.random.uniform(0, 2 * np.pi)
            
            # Step 1: Stand up (90° around Z)
            q_base = np.array([0.0, 0.0, 0.707, 0.707])
            
            # Step 2: Random yaw around X (additional rotation)
            q_yaw = euler_to_quaternion(random_yaw, 0, 0)
            
            # Combine: stand up then yaw
            rot_array = quaternion_multiply(q_base, q_yaw)
            rot = tuple(rot_array)
    else:
        # Random yaw
        # random_yaw = np.random.uniform(0, 2 * np.pi)
        # rot = euler_to_quaternion(0, 0, random_yaw)
        rot = (0.0, 0.0, 0.0, 1.0)

    rigid_obj = RigidObjectCfg(
        prim_path=prim_path,
        spawn=shape,
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(0.0, 0.0, 0.0) if fix_shape not in ["cross", "L", "t"] else pos,
            pos=(0.0, 0.0, 0.0),
            rot=rot,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )
    return rigid_obj

def quaternion_multiply(q1, q2):
    """Multiply two quaternions (x, y, z, w format)"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def euler_to_quaternion(roll, pitch, yaw):
    """Convert euler angles (in radians) to quaternion (x, y, z, w)"""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([x, y, z, w])


def get_mat_material_paths() -> list[str]:
    """
    Returns a curated list of standard NVIDIA MDL material paths 
    for the Visual Perspective Taking (VPT) task.
    """
    # base = f"{NVIDIA_NUCLEUS_DIR}/Materials/2023_1/vMaterials_2/"
    base = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base"

    material_paths = [
        f"{base}/Architecture/Ceiling_Tiles.mdl",
        f"{base}/Architecture/Roof_Tiles.mdl",
        f"{base}/Architecture/Shingles_01.mdl",
        f"{base}/Carpet/Carpet_Berbet_Gray.mdl",
        f"{base}/Carpet/Carpet_Diamond_Olive.mdl",
        f"{base}/Carpet/Carpet_Diamond_Yellow.mdl",
        f"{base}/Carpet/Carpet_Pattern_Leaf_Squares_Tan.mdl",
        f"{base}/Carpet/Carpet_Pattern_Squares_Multi.mdl",
        f"{base}/Carpet/Carpet_Forest.mdl",
        f"{base}/Carpet/Carpet_Cream.mdl",
        f"{base}/Carpet/Carpet_Beige.mdl",
        f"{base}/Carpet/Carpet_Berber_Multi.mdl",
        f"{base}/Masonry/Adobe_Brick.mdl",
        f"{base}/Masonry/Brick_Pavers.mdl",
        f"{base}/Masonry/Brick_Wall_Brown.mdl",
        f"{base}/Masonry/Brick_Wall_Red.mdl",
        f"{base}/Masonry/Concrete_Block.mdl",
        f"{base}/Masonry/Concrete_Rough.mdl",
        f"{base}/Masonry/Concrete_Formed.mdl",
        f"{base}/Metals/CorrugatedMetal.mdl",
        f"{base}/Metals/Brushed_Antique_Copper.mdl",
        f"{base}/Natural/Asphalt.mdl",
        f"{base}/Natural/Dirt.mdl",
        f"{base}/Natural/Grass_Cut.mdl",
        f"{base}/Natural/Grass_Countryside.mdl",
        f"{base}/Natural/Grass_Winter.mdl",
        f"{base}/Natural/Mulch_Brown.mdl",
        f"{base}/Natural/Soil_Rocky.mdl",
        f"{base}/Natural/Sand.mdl",
        f"{base}/Natural/Leaves.mdl",
        f"{base}/Plastics/Veneer_OU_Walnut.mdl",
        f"{base}/Plastics/Veneer_UX_Walnut_Cherry.mdl",
        f"{base}/Stone/Adobe_Octagon_Dots.mdl",
        f"{base}/Stone/Fieldstone.mdl",
        f"{base}/Stone/Pea_Gravel.mdl",
        f"{base}/Stone/Gravel_River_Rock.mdl",
        f"{base}/Stone/Gravel.mdl",
        f"{base}/Stone/Stone_Wall.mdl",
        f"{base}/Stone/Retaining_Block.mdl",
        f"{base}/Stone/Terracotta.mdl",
        f"{base}/Wood/Ash.mdl",
        f"{base}/Wood/Ash_Planks.mdl",
        f"{base}/Wood/Bamboo.mdl",
        f"{base}/Wood/Bamboo_Planks.mdl",
        f"{base}/Wood/Birch.mdl",
        f"{base}/Wood/Cardboard.mdl",
        f"{base}/Wood/Oak.mdl",
        f"{base}/Wood/Oak_Planks.mdl",
        f"{base}/Wood/Cherry.mdl",
        f"{base}/Wood/Cherry_Planks.mdl",
        f"{base}/Wood/Mahogany.mdl",
        f"{base}/Wood/Parquet_Floor.mdl",
        f"{base}/Wood/Plywood.mdl",
        f"{base}/Wood/Timber.mdl",
        f"{base}/Wood/Plaster.mdl",
        f"{base}/Wood/Walnut.mdl",
        f"{base}/Wood/Walnut_Planks.mdl",
    ]
    
    return material_paths

def get_vpt_material_paths() -> list[str]:
    """
    Returns a curated list of standard NVIDIA MDL material paths 
    for the Visual Perspective Taking (VPT) task.
    """
    base = f"{NVIDIA_NUCLEUS_DIR}/Materials/2023_1/vMaterials_2/"
    # base = f"{ISAAC_NUCLEUS_DIR}/Materials/Base"
    
    # Iterate through various subdir and get all .mdl files
    subdirs = ["Leather", "Carpet", "Masonry", "Metal/Mesh", "Paper", "Wood", "Leather"]

    # OS walk through all {base}/{subdir} and get all .mdl files
    material_paths = []
    for subdir in subdirs:
        full_subdir = os.path.join(base, subdir)
        for root, dirs, files in os.walk(full_subdir):
            for file in files:
                if file.endswith(".mdl"):
                    material_paths.append(os.path.join(root, file))
    
    return material_paths
