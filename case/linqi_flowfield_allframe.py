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
from skimage import io

from natsort import natsorted
# from particle_vtools.Explorer3D import Explorer3D
from particle_vtools.PoreStructure import PoreStructure_CT
from particle_vtools.FluidStructure import FluidIterator_CT

from pore_net.utils import(
    interpolate_velocity_field
)
# from particle_vtools.Particle import ParticleIterator_DF
# import argparse

save_fig = True
num_frames = 100


cmap = 'jet'
opacity = 0.5

tif_path = "/Users/chunyang/Downloads/lq/whole3images/image_3d.tif"

# Early's version
particle_pred_df_path = "/Users/chunyang/Downloads/lq/whole3images/interpore_inference_autoregressive_49_to_58.csv"  # noqa
particle_ground_df_path = "/Users/chunyang/Downloads/lq/whole3images/matched_particles_real_data_50_to_58.csv"  # noqa

# Second version - more particles
particle_pred_df_path = "/Users/chunyang/Downloads/lq/whole3images/interpore_predicted_R1.csv"  # noqa
particle_ground_df_path = "/Users/chunyang/Downloads/lq/whole3images/matched_real_R1.csv"  # noqa

# Fno's
tif_path = "/Users/chunyang/projects/particle/linqi/fno_8_1/image_3d.tif"
particle_pred_df_path = "/Users/chunyang/projects/particle/linqi/fno_8_1/fno_single_step_predictions.csv"  # noqa
particle_ground_df_path = "/Users/chunyang/projects/particle/linqi/fno_8_1/fno_single_step_predictions.csv"  # noqa


def process_frame(
    df,
    grid_size,
    x_key,
    y_key,
    z_key,
    vx_key,
    vy_key,
    vz_key,
    offset=[50, 50, 0],
):
    """
    Compute the physics grid for a single frame.

    The grid has 3 channels (for vx, vy, vz) and a spatial shape of grid_size.
    It uses the logic from your dataset method:
      - x_idx and y_idx are computed as (value + 50)
      - z_idx is computed as (value + 50) and then subtracted by 100.

    Parameters:
        df (pd.DataFrame): Dataframe containing the velocity data.
        frame: Value identifying the current frame.
        grid_size (tuple): Spatial grid size (D, H, W).
        x_key, y_key, z_key, frame_key (str): Column names for coordinates and
        frame. vx_key, vy_key, vz_key (str): Column names for
        the velocity components.

    Returns:
        numpy.ndarray: Grid of shape (3, D, H, W) with vx, vy, vz placed at
        the indexed locations.
    """
    off_x, off_y, off_z = offset
    grid = np.zeros((3,) + grid_size, dtype=np.float32)
    dff = df

    dff[f"{x_key}_idx"] = (dff[x_key]).astype(int)
    dff[f"{y_key}_idx"] = (dff[y_key]).astype(int)
    dff[f"{z_key}_idx"] = (dff[z_key]).astype(int)

    z_idx = dff[f"{z_key}_idx"].values + off_z
    y_idx = dff[f"{y_key}_idx"].values + off_y
    x_idx = dff[f"{x_key}_idx"].values + off_x

    grid[0, z_idx, y_idx, x_idx] = dff[vx_key].values
    grid[1, z_idx, y_idx, x_idx] = dff[vy_key].values
    grid[2, z_idx, y_idx, x_idx] = dff[vz_key].values
    return grid


pore_tif = io.imread(tif_path)

rock = PoreStructure_CT(
    tif_path, down_sample_factor=1, permute_axes=[2, 1, 0])
rock_mesh = rock.get_surface()
# pore_tif = pore_tif.transpose(2, 1, 0)
pore_mask = pore_tif // 255
grid_size = pore_tif.shape

pred_df = pd.read_csv(particle_pred_df_path)
gt_df = pd.read_csv(particle_ground_df_path)

# pred_cube = process_frame(
#     pred_df,
#     grid_size,
#     "x",
#     "y",
#     "z",
#     "vx_pred",
#     "vy_pred",
#     "vz_pred",
#     offset=[0, 0, 0]
# )

# gt_cube = process_frame(
#     gt_df,
#     grid_size,
#     "x",
#     "y",
#     "z",
#     "vx",
#     "vy",
#     "vz",
#     offset=[0, 0, 0]
# )

pred_cube = process_frame(
    pred_df,
    grid_size,
    "x_pred",
    "y_pred",
    "z_pred",
    "vx_pred",
    "vy_pred",
    "vz_pred",
    offset=[0, 0, 0]
)

gt_cube = process_frame(
    gt_df,
    grid_size,
    "x_true",
    "y_true",
    "z_true",
    "vx_true",
    "vy_true",
    "vz_true",
    offset=[0, 0, 0]
)

print("Interpolating velocity field, this may take a while...")
pred_cube = interpolate_velocity_field(
    pred_cube, pore_mask
)
print("interpolation for prediction done")

gt_cube = interpolate_velocity_field(
    gt_cube, pore_mask
)
print("interpolation for ground truth done")

pred_cube = np.linalg.norm(pred_cube, axis=0)
gt_cube = np.linalg.norm(gt_cube, axis=0)

pred_cube = pred_cube.transpose(2, 1, 0)
gt_cube = gt_cube.transpose(2, 1, 0)

# Save the velocity magnitude fields
np.save('gt_velocity_magnitude.npy', gt_cube)
np.save('pred_velocity_magnitude.npy', pred_cube)
print("Saved velocity magnitude fields")


p = pv.Plotter(
    title="Particle Prediction vs Ground Truth",
    shape=(1, 2),
    window_size=[2000, 1000])

# Window 1 - Ground Truth
p.subplot(0, 0)
# p.show_grid(
#     all_edges=True,
#     show_xlabels=False,
#     show_ylabels=False,
#     show_zlabels=False,
# )
p.add_text("Ground Truth", font_size=20)
# Calculate global min/max for consistent color scaling
# Calculate 5th and 95th percentiles across both arrays
global_min = np.percentile(
    np.concatenate([gt_cube.flatten(), pred_cube.flatten()]), 1)
global_max = np.percentile(
    np.concatenate([gt_cube.flatten(), pred_cube.flatten()]), 99.99)

print(f"Global min: {global_min:.2f}, Global max: {global_max:.2f}")
print(f"Prediction min: {pred_cube.min():.2f}, Prediction max: {pred_cube.max():.2f}")  # noqa
print(f"Ground truth min: {gt_cube.min():.2f}, Ground truth max: {gt_cube.max():.2f}")  # noqa


# Add the volume to the plotter
p.add_volume(
    gt_cube,
    opacity="linear",
    cmap="jet",
    clim=[global_min, global_max]
)

# Window 2 - Prediction
p.subplot(0, 1)
# p.show_grid(
#     all_edges=True,
#     show_xlabels=False,
#     show_ylabels=False,
#     show_zlabels=False,
# )
p.add_text("Prediction", font_size=20)
# Add the volume to the plotter
p.add_volume(
    pred_cube,
    opacity="linear",
    cmap="jet",
    clim=[global_min, global_max]
)
p.link_views()

p.camera_position = "yz"
p.camera.azimuth = -30
p.camera.elevation = 15

file_name = particle_pred_df_path.split(".")[0]

if not save_fig:
    p.show()

elif save_fig:
    p.camera.azimuth = 0
    p.open_gif(f"{file_name}_flowfield.gif", fps=15)
    p.camera.zoom(1.2)
    for i in range(num_frames):
        p.camera.azimuth = p.camera.azimuth - 1
        p.write_frame()
    p.close()
