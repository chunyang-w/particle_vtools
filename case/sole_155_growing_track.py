"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Visualization script to display growing particle trajectories over time
alongside the evolving fluid surface interface.

The script supports:
1. Interactive mode with a time slider to control trajectory growth
2. Render mode to export animated GIF showing temporal evolution
"""

import glob
import time
import numpy as np
import pandas as pd
import pyvista as pv

from natsort import natsorted
from particle_vtools.FluidStructure import FluidIterator_CT

# ============================================================================
# Configuration Parameters
# ============================================================================

# Render mode: True to save GIF, False for interactive mode
save_fig = False
num_frames = 90  # Number of frames for GIF output

# Visualization parameters
drop_percent = 0  # Percentage of particles to drop (0 = show all)
clim_low = 0.2  # Lower quantile for velocity colormap
clim_high = 0.95  # Upper quantile for velocity colormap

down_sample_factor = 10  # Downsampling factor for surface mesh
scale = 1

cmap = 'jet'
line_width = 3
opacity = 0.5

# Particle filtering
particle_offset = [0, 0, 0]
particle_max_height = 1180

show_bar = False  # Show/hide scalar bar

# Data paths
particle_ground_df_path = "/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv"  # noqa
ct_files_path = "/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped_interpolated/*"  # noqa

# Frame range for visualization
frame_start = 100
frame_end = 190

# ============================================================================
# Load Data
# ============================================================================

# Load CT files for fluid surface
ct_files = glob.glob(ct_files_path)
ct_files = natsorted(ct_files)
print(f"Total CT files found: {len(ct_files)}")

# Note: CT files are at half the temporal resolution of particle data
# So we need to map particle frames to CT file indices
ct_files = ct_files[frame_start:frame_end+1]
print(f"Using {len(ct_files)} CT files for frames {frame_start} to {frame_end}")

# Initialize fluid iterator
oil_iterator = FluidIterator_CT(
    "oil",
    ct_files,
    threshold=1,
    scale=scale,
    permute_axes=(2, 1, 0),
    down_sample_factor=down_sample_factor,
)

# Load particle trajectory data
ground_df = pd.read_csv(particle_ground_df_path)
print(f"Loaded {len(ground_df)} particle data points")


# ============================================================================
# Helper Functions
# ============================================================================

def get_track(
        df,
        x_key='x',
        y_key='y',
        z_key='z',
        vx_key='vx',
        vy_key='vy',
        vz_key='vz',
        frame_key='frame',
        particle_key="particle",
        drop_percent=0.0,
        frame_start=0,
        frame_end=1000,
        particle_offset=[0, 0, 0],
        particle_idx=None,
        ):
    """
    Generate PyVista PolyData for particle trajectories.
    
    This function creates trajectory lines from particle tracking data,
    always rebuilding from scratch to support bidirectional time navigation.
    """
    # Sort by particle ID and frame
    df = df.sort_values([particle_key, frame_key])

    # Randomly select subset of particles if drop_percent > 0
    keep_frac = (100 - drop_percent) / 100
    unique_ids = df[particle_key].unique()
    selected_ids = np.random.choice(
        unique_ids,
        size=int(len(unique_ids) * keep_frac),
        replace=False)
    df = df[df[particle_key].isin(selected_ids)]
    
    # Filter by specific particle indices if provided
    if particle_idx is not None:
        df = df[df[particle_key].isin(particle_idx)]
    
    # Filter by frame range and height
    df = df[(df[frame_key] >= frame_start) & (df[frame_key] <= frame_end)]
    df = df[df[z_key] < particle_max_height]

    # Build trajectory lines
    # Handle empty case early
    if len(df) == 0:
        poly = pv.PolyData()
        return poly
    
    # Extract all points and velocities at once (much faster than loop)
    all_points = df[[x_key, y_key, z_key]].values + np.array(particle_offset)
    
    # Calculate velocity magnitude for all points at once
    velocity_vectors = df[[vx_key, vy_key, vz_key]].values
    all_velocities = np.linalg.norm(velocity_vectors, axis=1)
    
    # Create cell connectivity efficiently using vectorized operations
    # Count points per particle
    particle_counts = df.groupby(particle_key, sort=False).size().values
    
    # Build cells array efficiently
    total_cells_size = np.sum(particle_counts + 1)  # +1 for count prefix
    cells = np.empty(total_cells_size, dtype=np.int64)
    
    cell_idx = 0
    point_idx = 0
    for n_points in particle_counts:
        # Insert count
        cells[cell_idx] = n_points
        # Insert point indices
        cells[cell_idx + 1:cell_idx + 1 + n_points] = np.arange(point_idx, point_idx + n_points)
        cell_idx += n_points + 1
        point_idx += n_points

    # Create PyVista PolyData object
    poly = pv.PolyData()
    poly.points = all_points
    poly.lines = cells
    poly['velocity'] = all_velocities
    
    return poly


# ============================================================================
# Calculate velocity range for consistent colormap
# ============================================================================

# Get full trajectory to determine velocity range
track_full = get_track(
    df=ground_df,
    x_key='x',
    y_key='y',
    z_key='z',
    vx_key='vx',
    vy_key='vy',
    vz_key='vz',
    frame_key='frame',
    particle_key='particle',
    frame_start=frame_start,
    frame_end=frame_end,
    drop_percent=drop_percent,
    particle_offset=particle_offset,
    particle_idx=None,
)

# Calculate velocity limits for consistent colormap across all frames
if len(track_full.points) > 0:
    velocity_min = np.quantile(track_full['velocity'], clim_low)
    velocity_max = np.quantile(track_full['velocity'], clim_high)
else:
    velocity_min = 0
    velocity_max = 1

print(f"Velocity range: {velocity_min:.3f} to {velocity_max:.3f}")


# ============================================================================
# Shared Scene Update Function
# ============================================================================

def create_scene_updater(plotter, mode='interactive'):
    """
    Create a scene update function that works for both interactive and render modes.
    Uses in-place mesh updates instead of removing/re-adding actors.
    
    Args:
        plotter: PyVista plotter instance
        mode: 'interactive' or 'render'
    
    Returns:
        update_scene function
    """
    # State tracking - store mesh objects and actors
    track_mesh = None
    track_actor = None
    surface_mesh = None
    surface_actor = None
    surface_wireframe_actor = None
    frame_text_actor = None
    
    def update_scene(current_frame):
        """Update scene to show trajectories up to current_frame."""
        nonlocal track_mesh, track_actor, surface_mesh, surface_actor
        nonlocal surface_wireframe_actor, frame_text_actor
        
        current_frame = int(current_frame)
        
        # Generate trajectories up to current frame
        track = get_track(
            df=ground_df,
            x_key='x',
            y_key='y',
            z_key='z',
            vx_key='vx',
            vy_key='vy',
            vz_key='vz',
            frame_key='frame',
            particle_key='particle',
            frame_start=frame_start,
            frame_end=current_frame,  # Grow to current frame
            drop_percent=drop_percent,
            particle_offset=particle_offset,
            particle_idx=None,
        )
        
        # Get fluid surface for current frame
        ct_idx = min(current_frame - frame_start, len(ct_files) - 1)
        surface = oil_iterator.get_surface(ct_idx)
        
        # Update or create trajectory mesh
        if track_actor is None or len(track.points) == 0:
            # First time or empty track - add new mesh
            if track_actor is not None:
                plotter.remove_actor(track_actor)
            if len(track.points) > 0:
                track_mesh = track
                track_actor = plotter.add_mesh(
                    track_mesh,
                    scalars='velocity',
                    line_width=line_width,
                    cmap=cmap,
                    render_lines_as_tubes=True,
                    opacity=opacity,
                    clim=[velocity_min, velocity_max],
                )
        else:
            # Update existing mesh in-place by setting properties
            track_mesh.points = track.points
            track_mesh.lines = track.lines
            track_mesh['velocity'] = track['velocity']
        
        # Update or create surface wireframe
        if surface_wireframe_actor is None:
            # First time - add new mesh
            surface_mesh = surface
            surface_wireframe_actor = plotter.add_mesh(
                surface_mesh,
                style='wireframe',
                line_width=0.9,
                color="blue",
                pbr=True,
                metallic=0.1,
                roughness=0.01,
                diffuse=1,
                opacity=0.05)
        else:
            # Update existing mesh in-place by setting properties
            surface_mesh.points = surface.points
            if hasattr(surface, 'faces'):
                surface_mesh.faces = surface.faces
        
        # # Remove scalar bar if requested
        # if not show_bar:
        #     try:
        #         plotter.remove_scalar_bar()
        #     except (StopIteration, KeyError):
        #         pass
        
        # # Update frame counter text
        # if frame_text_actor is not None:
        #     plotter.remove_actor(frame_text_actor)
        # frame_text_actor = plotter.add_text(
        #     f"Frame: {current_frame}",
        #     position="upper_right",
        #     font_size=14,
        #     name="frame_text")
        print(f"Frame: {current_frame}")
        return
        # return track_actor, surface_actor, surface_wireframe_actor
    
    return update_scene


# ============================================================================
# Interactive Mode with Time Slider
# ============================================================================

if not save_fig:
    # Initialize plotter
    p = pv.Plotter(
        title="Growing Particle Trajectories - 155 Dataset",
        window_size=[1250, 1250])
    
    # Create scene updater for interactive mode
    update_scene = create_scene_updater(p, mode='interactive')
    
    # Initialize scene with starting frame
    update_scene(frame_start)
    
    # Add time slider
    p.add_slider_widget(
        update_scene,
        [frame_start, frame_end],
        value=frame_start,
        title="Time Frame",
        pointa=(0.1, 0.9),
        pointb=(0.4, 0.9),
        style='modern',
    )
    
    # Add title
    p.add_text(
        "Growing Particle Trajectories",
        position="upper_left",
        font_size=20)
    
    # Set camera position
    p.camera_position = "yz"
    p.camera.azimuth = -120
    p.camera.elevation = 10
    p.camera.zoom(1.2)
    
    # Show interactive window
    p.show()


# ============================================================================
# Render Mode - Export GIF
# ============================================================================

elif save_fig:
    # Initialize plotter - try offscreen first, fallback to window if needed
    p = pv.Plotter(
        title="Growing Particle Trajectories - 155 Dataset",
        window_size=[1250, 1250],
        # off_screen=True,
        )

    # Set camera position
    p.camera_position = "yz"
    p.camera.azimuth = -120
    p.camera.elevation = 10
    p.camera.zoom(1.2)

    # Set output filename
    run_name = f"out/sole_155_growing_track_f{frame_start}-{frame_end}"
    
    # Open GIF writer
    p.open_gif(f"{run_name}.gif", fps=4)
    
    # Calculate frame indices for animation
    # Use actual frame range for better quality
    total_frames = frame_end - frame_start + 1
    
    # Adjust num_frames if it exceeds available frames to avoid repetition
    actual_num_frames = min(num_frames, total_frames)
    
    if actual_num_frames < num_frames:
        print(f"Note: Adjusted num_frames from {num_frames} to {actual_num_frames} to match available data")
    
    # Sample frames evenly across the range
    if actual_num_frames == total_frames:
        # Use all frames
        frame_indices = np.arange(frame_start, frame_end + 1)
    else:
        # Sample evenly
        frame_indices = np.linspace(frame_start, frame_end, actual_num_frames, dtype=int)
        # Remove duplicates while preserving order
        frame_indices = np.array(sorted(set(frame_indices)))
        actual_num_frames = len(frame_indices)
    
    print(f"Rendering {actual_num_frames} frames from time {frame_start} to {frame_end}...")
    
    # Create scene updater for render mode
    update_scene = create_scene_updater(p, mode='interactive')
    
    # Add persistent title (only once, not per frame)
    p.add_text(
        "Growing Particle Trajectories",
        position="upper_left",
        font_size=20)
    
    for idx, current_frame in enumerate(frame_indices):
        print(f"Rendering frame {idx+1}/{actual_num_frames} (time frame {current_frame})")
        
        # Update scene using shared function (updates meshes in-place)
        update_scene(current_frame)
        # time.sleep(4)
        
        # Write frame to GIF
        p.write_frame()
    
    # Close GIF writer
    p.close()

    print(f"Done! Output saved to:")
    print(f"  - {run_name}.gif")

