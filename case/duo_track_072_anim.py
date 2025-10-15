"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Visulisation script utilising pyvista API to compare the particle tracking

A 3D scene with two subplots is created to compare the particle tracking
between the ground truth and the prediction.

Two view are linked to facilitate the comparison.
"""

import glob
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

from natsort import natsorted
from tqdm import tqdm

# from particle_vtools.Explorer3D import Explorer3D
# from particle_vtools.PoreStructure import PoreStructure_CT
from particle_vtools.FluidStructure import FluidIterator_CT
# from particle_vtools.Particle import ParticleIterator_DF
# import argparse

save_fig = True
num_frames = 100

drop_percent = 0
clim_low = 0.2
clim_high = 0.95

down_sample_factor = 8
scale = 1
show_surface = True
surface_idx = 29

cmap = 'jet'
line_width = 3
opacity = 0.5


# frame_start = 150
# frame_end = frame_start + 30

particle_offset = [50, 50, -460]

frame_start = 100
frame_end = frame_start + 80
# particle_pred_df_path = "/Users/chunyang/Downloads/072_autoregressive_noise5_predictions.csv"  # noqa
# particle_pred_df_path = "/Users/chunyang/Downloads/72_t150-180_partial_073_teston_072.csv"  # noqa
# particle_pred_df_path = "/Users/chunyang/Downloads/72_t150-180.csv"  # noqa
particle_pred_df_path = "/Users/chunyang/Downloads/72_t100-180.csv"  # noqa  long rollout 072
# particle_pred_df_path = "/Users/chunyang/Downloads/72_t100-180 (1).csv"  # noqa long rollout 072 with acc
particle_ground_df_path = "/Users/chunyang/projects/particle/data/Velocity_smooth/072_final.csv"  # noqa
ct_files_path = "/Users/chunyang/projects/particle/data/Segmentations/072/*"  # noqa


# Load the oil surface
ct_files = glob.glob(ct_files_path) # noqa
ct_files = natsorted(ct_files)
ct_files = ct_files[frame_start//2:frame_end//2+1]
ct_files = [[f, f] for f in ct_files]
ct_files = [item for sublist in ct_files for item in sublist]
print("jiji")
print("len(ct_files):", len(ct_files))
print(ct_files)

oil_iterator = FluidIterator_CT(
    "oil",
    ct_files,
    threshold=255,
    scale=scale,
    permute_axes=(2, 1, 0),
    down_sample_factor=down_sample_factor,
)


def calculate_r_squared(pred, true):
    """Calculate R-squared between predicted and true values."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def calculate_rmse(pred, true):
    """Calculate Root Mean Square Error between predicted and true values.
    Args:
        pred: Predicted values (numpy array or pandas series)
        true: True/ground truth values (numpy array or pandas series)
    Returns:
        float: RMSE value
    """
    # Convert to numpy arrays if they aren't already
    pred = np.array(pred)
    true = np.array(true)
    # Calculate RMSE
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def get_track(
        df,
        x_key='pred_x',
        y_key='pred_y',
        z_key='pred_z',
        vx_key='pred_vx',
        vy_key='pred_vy',
        vz_key='pred_vz',
        frame_key='frame',
        particle_key="particle_id",
        drop_percent=0.0,
        frame_start=0,
        frame_end=1000,
        particle_offset=[0, 0, 0],
        particle_idx=None,
        ):
    # Create trajectories for each particle
    # print(f"Getting track for frame {frame_start} to {frame_end}")
    df = df.sort_values([particle_key, frame_key])

    # Randomly select subset of particles
    keep_frac = (100 - drop_percent)/100
    unique_ids = df[particle_key].unique()
    selected_ids = np.random.choice(
            unique_ids,
            size=int(len(unique_ids)*keep_frac),
            replace=False)
    df = df[df[particle_key].isin(selected_ids)]
    if particle_idx is not None:
        df = df[df[particle_key].isin(particle_idx)]
    df = df[(df[frame_key] >= frame_start) & (df[frame_key] <= frame_end)]

    # Vectorized approach - extract all data at once
    all_points = df[[x_key, y_key, z_key]].values + np.array(particle_offset)
    all_velocities = np.linalg.norm(
        df[[vx_key, vy_key, vz_key]].values, axis=1)

    # Get particle group sizes efficiently
    particle_groups = df.groupby(particle_key, sort=False)
    particle_counts = particle_groups.size().values

    # Fully vectorized cell connectivity creation
    n_particles = len(particle_counts)

    # Create index arrays for all cells at once
    total_points = particle_counts.sum()
    total_cells_size = total_points + n_particles

    # Create the cells array using vectorized operations
    cells = np.empty(total_cells_size, dtype=np.int64)

    # Calculate positions where count values go
    count_positions = np.zeros(n_particles, dtype=np.int64)
    count_positions[1:] = np.cumsum(particle_counts[:-1] + 1)

    # Insert particle counts at their positions
    cells[count_positions] = particle_counts

    # Create mask for non-count positions
    mask = np.ones(total_cells_size, dtype=bool)
    mask[count_positions] = False

    # Fill in the point indices
    point_indices = np.arange(total_points)
    cells[mask] = point_indices

    # Create PyVista object
    poly = pv.PolyData()
    poly.points = all_points
    poly.lines = cells
    poly['velocity'] = all_velocities
    return poly, df


# Fluid surface
surface = oil_iterator.get_surface(surface_idx)


pred_df = pd.read_csv(particle_pred_df_path)
ground_df = pd.read_csv(particle_ground_df_path)
selected_idx = pred_df["particle"].unique()

track_pred_list = []
track_ground_list = []
rmse_list = []
r_squared_list = []

for i in tqdm(range(frame_start, frame_end)):
    track_pred, df_pred = get_track(
        df=pred_df,
        x_key='x',
        y_key='y',
        z_key='z',
        vx_key='vx',
        vy_key='vy',
        vz_key='vz',
        frame_key='frame',
        particle_key='particle',
        frame_start=frame_start,
        frame_end=i,
        drop_percent=drop_percent,
        particle_offset=particle_offset,
        particle_idx=selected_idx,
    )

    track_ground, df_ground = get_track(
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
        frame_end=i,
        drop_percent=drop_percent,
        particle_offset=particle_offset,
        particle_idx=selected_idx,
    )
    track_pred_list.append(track_pred)
    track_ground_list.append(track_ground)

    df_merged = pd.merge(
        df_pred, df_ground,
        on=['particle', 'frame'],
        suffixes=('_pd', '_gt')
    )

    # Calculate velocity magnitudes for exactly matched data
    rmse_x = calculate_rmse(df_merged['x_pd'], df_merged['x_gt'])
    rmse_y = calculate_rmse(df_merged['y_pd'], df_merged['y_gt'])
    rmse_z = calculate_rmse(df_merged['z_pd'], df_merged['z_gt'])
    rmse = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)
    rmse_list.append(rmse)
    r_squared_x = calculate_r_squared(df_merged['x_pd'], df_merged['x_gt'])
    r_squared_y = calculate_r_squared(df_merged['y_pd'], df_merged['y_gt'])
    r_squared_z = calculate_r_squared(df_merged['z_pd'], df_merged['z_gt'])
    r_squared_mean = np.mean([r_squared_x, r_squared_y, r_squared_z])
    r_squared_list.append(r_squared_mean)
    print(f"RMSE: {rmse} \t R-squared: {r_squared_mean}")

p = pv.Plotter(
    title="Particle Prediction vs Ground Truth",
    shape=(1, 2),
    window_size=[2500, 1250])

# Window 1 - Ground Truth
# p.show_grid(
#     all_edges=True,
#     show_xlabels=False,
#     show_ylabels=False,
#     show_zlabels=False,
# )
p.add_text("Ground Truth", font_size=20)
actor_gt = p.add_mesh(
    track_ground_list[0],
    scalars='velocity',
    line_width=line_width,
    cmap=cmap,
    render_lines_as_tubes=True,
    opacity=opacity,
    clim=[np.quantile(track_ground['velocity'], clim_low),
          np.quantile(track_ground['velocity'], clim_high)],
)
p.add_mesh(
    surface,
    color="blue",
    pbr=True,
    opacity=0.01)

p.add_mesh(
    surface,
    style='wireframe',
    line_width=0.9,
    color="blue",
    pbr=True,
    metallic=0.1,
    roughness=0.01,
    diffuse=1,
    opacity=0.03)

# Window 2 - Prediction
p.subplot(0, 1)
# p.show_grid(
#     all_edges=True,
#     show_xlabels=False,
#     show_ylabels=False,
#     show_zlabels=False,
# )
p.add_text("Prediction", font_size=20)
actor_pd = p.add_mesh(
    track_pred_list[0],
    scalars='velocity',
    line_width=line_width,
    cmap=cmap,
    render_lines_as_tubes=True,
    opacity=opacity,
    clim=[np.quantile(track_ground['velocity'], clim_low),
          np.quantile(track_ground['velocity'], clim_high)],
)
p.add_mesh(
    surface,
    color="blue",
    pbr=True,
    opacity=0.01)

p.add_mesh(
    surface,
    style='wireframe',
    line_width=0.9,
    color="blue",
    pbr=True,
    metallic=0.1,
    roughness=0.01,
    diffuse=1,
    opacity=0.03)

text_actor = p.add_text(
    f"Frame: {0}",
    position='lower_right',
    font_size=24,
    color='black')


p.link_views()

p.camera_position = "yz"
p.camera.azimuth = -70
p.camera.elevation = 15
p.camera.zoom(1.5)


csv_base_name = particle_pred_df_path.split("/")[-1].split(".")[0]
run_name = f"out/duo_track_{csv_base_name}"

# Set matplotlib parameters for journal-quality plots
plt.rcParams.update({
    'font.size': 14,                # Base font size
    'axes.labelsize': 20,           # Axis label font size
    'axes.titlesize': 20,           # Title font size
    'xtick.labelsize': 18,          # X-axis tick label size
    'ytick.labelsize': 18,          # Y-axis tick label size
    'legend.fontsize': 18,          # Legend font size
    'figure.titlesize': 22,         # Figure title size
    'font.family': 'sans-serif',    # Font family
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],  # Fonts
    'axes.linewidth': 1.5,          # Axis line width
    'axes.grid': True,              # Show grid by default
    'grid.alpha': 0.3,              # Grid transparency
    'grid.linewidth': 0.8,          # Grid line width
    'lines.linewidth': 2.5,         # Line plot width
    'lines.markersize': 8,          # Marker size
    'xtick.major.width': 1.5,       # X-axis major tick width
    'ytick.major.width': 1.5,       # Y-axis major tick width
    'xtick.major.size': 6,          # X-axis major tick size
    'ytick.major.size': 6,          # Y-axis major tick size
    'figure.dpi': 100,              # Display DPI
    'savefig.dpi': 300,            # Save DPI for high-quality output
    'savefig.bbox': 'tight',       # Tight bounding box
    'savefig.pad_inches': 0.1      # Padding around saved figure
})

# RMSE Plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(len(rmse_list)), rmse_list,
        marker='o', markersize=8, linewidth=2.5,
        color='#1f77b4', markerfacecolor='#1f77b4',
        markeredgecolor='white', markeredgewidth=1.5)
ax.set_xlabel('Frame Number', fontsize=16, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=16, fontweight='bold')
ax.set_title('RMSE vs Frame Number', fontsize=18, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
plt.tight_layout()
plt.savefig(f"{run_name}_rmse.png", dpi=300, bbox_inches='tight',
            pad_inches=0.1)
# Also save as PDF for vector graphics
plt.savefig(f"{run_name}_rmse.pdf", dpi=300, bbox_inches='tight',
            pad_inches=0.1)
plt.show()

# R-squared Plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(len(r_squared_list)), r_squared_list,
        marker='s', markersize=8, linewidth=2.5,
        color='#2ca02c', markerfacecolor='#2ca02c',
        markeredgecolor='white', markeredgewidth=1.5)
ax.set_xlabel('Frame Number', fontsize=16, fontweight='bold')
ax.set_ylabel('R-squared', fontsize=16, fontweight='bold')
ax.set_title('R-squared vs Frame Number', fontsize=18, fontweight='bold',
             pad=15)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
plt.tight_layout()
plt.savefig(f"{run_name}_r_squared.png", dpi=300, bbox_inches='tight',
            pad_inches=0.1)
# Also save as PDF for vector graphics
plt.savefig(f"{run_name}_r_squared.pdf", dpi=300, bbox_inches='tight',
            pad_inches=0.1)
plt.show()


def update_scene(frame_idx):
    global actor_gt
    global actor_pd
    global text_actor
    p.subplot(0, 0)
    p.remove_actor(actor_gt)
    actor_gt = p.add_mesh(
        track_ground_list[frame_idx],
        scalars='velocity',
        line_width=line_width,
        cmap=cmap, render_lines_as_tubes=True,
        opacity=opacity,
        clim=[np.quantile(track_ground['velocity'], clim_low),
              np.quantile(track_ground['velocity'], clim_high)])
    p.subplot(0, 1)
    p.remove_actor(actor_pd)
    actor_pd = p.add_mesh(track_pred_list[frame_idx],
                          scalars='velocity',
                          line_width=line_width, cmap=cmap,
                          render_lines_as_tubes=True, opacity=opacity,
                          clim=[np.quantile(track_ground['velocity'], clim_low),
                                np.quantile(track_ground['velocity'], clim_high)])
    # Add frame index text in bottom right corner
    p.remove_actor(text_actor)
    text_actor = p.add_text(
        f"Frame: {frame_idx}",
        position='lower_right',
        font_size=24,
        color='black')
    # p.camera.azimuth = p.camera.azimuth - 1
    p.write_frame()


if not save_fig:
    p.show()

elif save_fig:
    # p.camera.azimuth = 0

    p.export_html(f"{run_name}.html")
    p.open_gif(f"{run_name}.gif", fps=20)
    for i in range(len(track_ground_list)):
        # p.camera.azimuth = p.camera.azimuth - 1
        print(f"Updating scene to frame {i}")
        update_scene(i)
        p.write_frame()
    p.close()
