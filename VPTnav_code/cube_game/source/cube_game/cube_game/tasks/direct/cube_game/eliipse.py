import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EllipseViewPlanner:
    """Helper class for positioning an agent on an ellipse with camera and target at foci."""
    
    def __init__(self, camera_pos: np.ndarray, target_pos: np.ndarray, 
                 a: float, b: float, fov_degrees: float = 34.0):
        """
        Initialize ellipse view planner.
        
        Args:
            camera_pos: 3D position of camera object (focus 1)
            target_pos: 3D position of target object (focus 2)
            a: Semi-major axis length
            b: Semi-minor axis length
            fov_degrees: Horizontal field of view in degrees
        """
        self.camera_pos = np.array(camera_pos)
        self.target_pos = np.array(target_pos)
        self.a = a
        self.b = b
        self.fov_rad = np.radians(fov_degrees)
        
        # Calculate center and focal distance
        self.center = (self.camera_pos + self.target_pos) / 2
        self.c = np.linalg.norm(self.target_pos - self.camera_pos) / 2
        
        # Calculate ellipse orientation (direction from camera to target)
        self.major_axis_dir = (self.target_pos - self.camera_pos) / (2 * self.c)
        
        # Create perpendicular direction for minor axis (in xy-plane)
        self.minor_axis_dir = np.array([-self.major_axis_dir[1], 
                                        self.major_axis_dir[0], 
                                        0])
        if np.linalg.norm(self.minor_axis_dir) > 0:
            self.minor_axis_dir /= np.linalg.norm(self.minor_axis_dir)
    
    def get_point_on_ellipse(self, theta: float) -> np.ndarray:
        """
        Get 3D position on ellipse for given angle parameter.
        
        Args:
            theta: Angle parameter in radians (0 to 2π)
            
        Returns:
            3D position on ellipse
        """
        x = self.a * np.cos(theta)
        y = self.b * np.sin(theta)
        
        # Transform to world coordinates
        pos = (self.center + 
               x * self.major_axis_dir + 
               y * self.minor_axis_dir)
        
        return pos
    
    def calculate_required_fov(self, agent_pos: np.ndarray) -> float:
        """
        Calculate the required FOV angle to see both foci from agent position.
        
        Args:
            agent_pos: 3D position of agent
            
        Returns:
            Required FOV in radians
        """
        # Vectors from agent to each focus
        to_camera = self.camera_pos - agent_pos
        to_target = self.target_pos - agent_pos
        
        # Calculate angle between them
        cos_angle = np.dot(to_camera, to_target) / (
            np.linalg.norm(to_camera) * np.linalg.norm(to_target)
        )
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return angle
    
    def is_valid_viewpoint(self, theta: float) -> bool:
        """
        Check if a point on the ellipse provides valid FOV coverage.
        
        Args:
            theta: Angle parameter in radians
            
        Returns:
            True if both foci fit in FOV from this position
        """
        pos = self.get_point_on_ellipse(theta)
        required_fov = self.calculate_required_fov(pos)
        return required_fov <= self.fov_rad
    
    def find_valid_arc_ranges(self, num_samples: int = 360) -> List[Tuple[float, float]]:
        """
        Find continuous arc ranges where FOV constraint is satisfied.
        
        Args:
            num_samples: Number of points to sample around ellipse
            
        Returns:
            List of (start_theta, end_theta) tuples for valid arcs
        """
        thetas = np.linspace(0, 2*np.pi, num_samples)
        valid = [self.is_valid_viewpoint(t) for t in thetas]
        
        # Find continuous ranges
        ranges = []
        start = None
        
        for i, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = thetas[i]
            elif not is_valid and start is not None:
                ranges.append((start, thetas[i-1]))
                start = None
        
        # Handle wraparound
        if start is not None:
            ranges.append((start, thetas[-1]))
        
        return ranges
    
    def get_lookat_direction(self, agent_pos: np.ndarray) -> np.ndarray:
        """
        Get the direction vector for agent to look at ellipse center.
        
        Args:
            agent_pos: 3D position of agent
            
        Returns:
            Normalized direction vector
        """
        direction = self.center - agent_pos
        return direction / np.linalg.norm(direction)
    
    def sample_valid_positions(self, num_positions: int = 8) -> List[np.ndarray]:
        """
        Sample valid agent positions around the ellipse.
        
        Args:
            num_positions: Number of positions to sample
            
        Returns:
            List of 3D positions
        """
        valid_ranges = self.find_valid_arc_ranges()
        
        if not valid_ranges:
            print("Warning: No valid viewpoints found!")
            return []
        
        # Calculate total valid arc length
        total_arc = sum(end - start for start, end in valid_ranges)
        
        positions = []
        for i in range(num_positions):
            # Sample proportionally across all valid arcs
            target_arc = (i / num_positions) * total_arc
            
            cumulative = 0
            for start, end in valid_ranges:
                arc_length = end - start
                if cumulative + arc_length >= target_arc:
                    theta = start + (target_arc - cumulative)
                    positions.append(self.get_point_on_ellipse(theta))
                    break
                cumulative += arc_length
        
        return positions
    
    def get_minimum_ellipse_params(self) -> Tuple[float, float]:
        """
        Calculate minimum a and b needed for full ellipse coverage.
        
        Returns:
            (a_min, b_min) tuple
        """
        half_fov = self.fov_rad / 2
        b_min = self.c / np.tan(half_fov)
        a_min = np.sqrt(b_min**2 + self.c**2)
        
        return a_min, b_min
    
    def plot_2d(self, num_samples: int = 360, show_valid_only: bool = True,
                figsize: Tuple[int, int] = (10, 8)):
        """
        Plot 2D top-down view of the ellipse with valid/invalid regions.
        
        Args:
            num_samples: Number of points to sample around ellipse
            show_valid_only: If True, only show valid arc segments
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate ellipse points
        thetas = np.linspace(0, 2*np.pi, num_samples)
        positions = np.array([self.get_point_on_ellipse(t) for t in thetas])
        
        # Check validity for each point
        valid_mask = np.array([self.is_valid_viewpoint(t) for t in thetas])
        
        # Plot ellipse
        if show_valid_only:
            # Plot valid segments in green, invalid in red
            ax.scatter(positions[valid_mask, 0], positions[valid_mask, 1], 
                      c='green', s=2, label='Valid viewpoints', alpha=0.6)
            ax.scatter(positions[~valid_mask, 0], positions[~valid_mask, 1], 
                      c='red', s=2, label='Invalid viewpoints', alpha=0.6)
        else:
            ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, label='Ellipse')
        
        # Plot foci (camera and target)
        ax.plot(self.camera_pos[0], self.camera_pos[1], 'ro', 
                markersize=10, label='Camera', zorder=5)
        ax.plot(self.target_pos[0], self.target_pos[1], 'bs', 
                markersize=10, label='Target', zorder=5)
        
        # Plot center
        ax.plot(self.center[0], self.center[1], 'k+', 
                markersize=15, label='Center', zorder=5)
        
        # Sample and plot some valid positions
        sample_positions = self.sample_valid_positions(num_positions=8)
        if sample_positions:
            sample_positions = np.array(sample_positions)
            ax.plot(sample_positions[:, 0], sample_positions[:, 1], 'mo', 
                   markersize=8, label='Sample positions', zorder=4)
            
            # Draw view lines from sample positions to center
            for pos in sample_positions:
                ax.plot([pos[0], self.center[0]], [pos[1], self.center[1]], 
                       'm--', alpha=0.3, linewidth=1)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Ellipse View Planning (FOV={np.degrees(self.fov_rad):.1f}°)\n'
                    f'a={self.a:.2f}, b={self.b:.2f}, c={self.c:.2f}')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_3d(self, num_samples: int = 360, show_valid_only: bool = True,
                figsize: Tuple[int, int] = (12, 9)):
        """
        Plot 3D view of the ellipse with valid/invalid regions.
        
        Args:
            num_samples: Number of points to sample around ellipse
            show_valid_only: If True, only show valid arc segments
            figsize: Figure size tuple
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate ellipse points
        thetas = np.linspace(0, 2*np.pi, num_samples)
        positions = np.array([self.get_point_on_ellipse(t) for t in thetas])
        
        # Check validity for each point
        valid_mask = np.array([self.is_valid_viewpoint(t) for t in thetas])
        
        # Plot ellipse
        if show_valid_only:
            ax.scatter(positions[valid_mask, 0], positions[valid_mask, 1], 
                      positions[valid_mask, 2], c='green', s=5, 
                      label='Valid viewpoints', alpha=0.6)
            ax.scatter(positions[~valid_mask, 0], positions[~valid_mask, 1], 
                      positions[~valid_mask, 2], c='red', s=5, 
                      label='Invalid viewpoints', alpha=0.6)
        else:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'b-', alpha=0.5, label='Ellipse')
        
        # Plot foci (camera and target)
        ax.scatter(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2], 
                  c='red', s=100, marker='o', label='Camera', zorder=5)
        ax.scatter(self.target_pos[0], self.target_pos[1], self.target_pos[2], 
                  c='blue', s=100, marker='s', label='Target', zorder=5)
        
        # Plot center
        ax.scatter(self.center[0], self.center[1], self.center[2], 
                  c='black', s=100, marker='+', label='Center', zorder=5)
        
        # Sample and plot some valid positions
        sample_positions = self.sample_valid_positions(num_positions=8)
        if sample_positions:
            sample_positions = np.array(sample_positions)
            ax.scatter(sample_positions[:, 0], sample_positions[:, 1], 
                      sample_positions[:, 2], c='magenta', s=80, 
                      marker='o', label='Sample positions', zorder=4)
            
            # Draw view lines from sample positions to center
            for pos in sample_positions:
                ax.plot([pos[0], self.center[0]], 
                       [pos[1], self.center[1]], 
                       [pos[2], self.center[2]], 
                       'm--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Ellipse View Planning (FOV={np.degrees(self.fov_rad):.1f}°)\n'
                    f'a={self.a:.2f}, b={self.b:.2f}, c={self.c:.2f}')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                             positions[:, 1].max()-positions[:, 1].min(),
                             positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
        mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_fov_requirement(self, num_samples: int = 360, 
                            figsize: Tuple[int, int] = (12, 6)):
        """
        Plot required FOV as a function of position around the ellipse.
        
        Args:
            num_samples: Number of points to sample around ellipse
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate ellipse points
        thetas = np.linspace(0, 2*np.pi, num_samples)
        required_fovs = []
        
        for t in thetas:
            pos = self.get_point_on_ellipse(t)
            required_fov = self.calculate_required_fov(pos)
            required_fovs.append(np.degrees(required_fov))
        
        required_fovs = np.array(required_fovs)
        
        # Plot required FOV
        ax.plot(np.degrees(thetas), required_fovs, 'b-', linewidth=2, 
               label='Required FOV')
        
        # Plot camera FOV limit
        ax.axhline(y=np.degrees(self.fov_rad), color='r', linestyle='--', 
                  linewidth=2, label=f'Camera FOV = {np.degrees(self.fov_rad):.1f}°')
        
        # Shade valid regions
        valid_mask = required_fovs <= np.degrees(self.fov_rad)
        ax.fill_between(np.degrees(thetas), 0, required_fovs, 
                        where=valid_mask, alpha=0.3, color='green', 
                        label='Valid region')
        ax.fill_between(np.degrees(thetas), 0, required_fovs, 
                        where=~valid_mask, alpha=0.3, color='red', 
                        label='Invalid region')
        
        ax.set_xlabel('Angle around ellipse (degrees)')
        ax.set_ylabel('Required FOV (degrees)')
        ax.set_title('FOV Requirement vs Position on Ellipse')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 360)
        
        # Add markers for special points
        special_angles = [0, 90, 180, 270]
        special_labels = ['Right (major)', 'Top (minor)', 'Left (major)', 'Bottom (minor)']
        for angle, label in zip(special_angles, special_labels):
            idx = int(angle / 360 * num_samples)
            ax.plot(angle, required_fovs[idx], 'ko', markersize=8)
            ax.annotate(f'{label}\n{required_fovs[idx]:.1f}°', 
                       xy=(angle, required_fovs[idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        return fig, ax


# Example usage
if __name__ == "__main__":
    # Define camera and target positions
    camera_pos = np.array([0, 0, 1])
    target_pos = np.array([2, 0, 1])
    
    # Create planner with smaller ellipse (won't guarantee full coverage)
    planner = EllipseViewPlanner(
        camera_pos=camera_pos,
        target_pos=target_pos,
        a=3.5,  # Semi-major axis
        b=3.3,  # Semi-minor axis
        fov_degrees=34
    )
    
    print(f"Focal distance c: {planner.c:.2f}")
    print(f"Current a: {planner.a:.2f}, b: {planner.b:.2f}")
    
    # Find minimum parameters for full coverage
    a_min, b_min = planner.get_minimum_ellipse_params()
    print(f"\nMinimum for full coverage:")
    print(f"a_min: {a_min:.2f}, b_min: {b_min:.2f}")
    
    # Find valid arc ranges
    valid_ranges = planner.find_valid_arc_ranges()
    print(f"\nValid arc ranges (radians):")
    for start, end in valid_ranges:
        print(f"  {start:.2f} to {end:.2f} ({np.degrees(end-start):.1f}°)")
    
    # Sample some valid positions
    positions = planner.sample_valid_positions(num_positions=8)
    print(f"\nSampled {len(positions)} valid positions")
    
    # Check a specific point
    theta = np.pi / 2  # Top of ellipse
    pos = planner.get_point_on_ellipse(theta)
    required_fov = planner.calculate_required_fov(pos)
    lookat = planner.get_lookat_direction(pos)
    
    print(f"\nAt theta={np.degrees(theta):.0f}°:")
    print(f"  Position: {pos}")
    print(f"  Required FOV: {np.degrees(required_fov):.1f}°")
    print(f"  Valid: {planner.is_valid_viewpoint(theta)}")
    print(f"  Look-at direction: {lookat}")
    
    # Visualize
    print("\nGenerating visualizations...")
    
    # 2D plot
    planner.plot_2d()
    
    # 3D plot
    planner.plot_3d()
    
    # FOV requirement plot
    planner.plot_fov_requirement()
    
    plt.show()