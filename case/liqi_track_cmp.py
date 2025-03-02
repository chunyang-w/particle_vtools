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

from natsort import natsorted

# from particle_vtools.Explorer3D import Explorer3D
# from particle_vtools.PoreStructure import PoreStructure_CT
from particle_vtools.FluidStructure import FluidIterator_CT
# from particle_vtools.Particle import ParticleIterator_DF
# import argparse

save_fig = True
num_frames = 100

drop_percent = 70
clim_low = 0.2
clim_high = 0.95

down_sample_factor = 8
scale = 2
show_surface = True
surface_idx = 0

cmap = 'jet'
line_width = 3
opacity = 0.5

# pore_tif_path = "../data/073_combined_results/073_segmentedTimeSteps_downsampledx2_tif/073_segmented_00000.tif"  # noqa
# ct_files_path = "../data/073_combined_results/073_segmentedTimeSteps_downsampledx2_tif/*"  # noqa
particle_pred_df_path = "/Users/chunyang/Downloads/test2.csv"  # noqa
particle_ground_df_path = "/Users/chunyang/Downloads/test2.csv"  # noqa
# ct_files_path = "../data/073_combined_results/073_segmentedTimeSteps_downsampledx2_tif/*"  # noqa

# Load the oil surface
# ct_files = glob.glob(ct_files_path) # noqa
# ct_files = natsorted(ct_files)
# ct_files = ct_files[85:106]
# oil_iterator = FluidIterator_CT(
#     "oil",
#     ct_files,
#     threshold=1,
#     scale=scale,
#     permute_axes=(2, 1, 0),
#     down_sample_factor=down_sample_factor,
#     )


def get_track(
        df,
        x_key='pred_x',
        y_key='pred_y',
        z_key='pred_z',
        vx_key='pred_vx',
        vy_key='pred_vy',
        vz_key='pred_vz',
        drop_percent=0.0,
        ):
    # Create trajectories for each particle
    df = df.sort_values(['particle', 'frame'])

    # Randomly select subset of particles
    keep_frac = (100 - drop_percent)/100
    unique_ids = df['particle'].unique()
    selected_ids = np.random.choice(
            unique_ids,
            size=int(len(unique_ids)*keep_frac),
            replace=False)
    df = df[df.particle.isin(selected_ids)]

    lines = []
    velocities = []
    particle_groups = df.groupby('particle')

    for pid, particle_data in particle_groups:
        # Extract ordered points for this particle
        points = particle_data[[x_key, y_key, z_key]].values
        lines.append(points)
        # Calculate velocity magnitude
        velocity = np.linalg.norm(
            particle_data[[vx_key, vy_key, vz_key]].values, axis=1)
        velocities.append(velocity)

    # Combine all data into PyVista-friendly format
    all_points = np.vstack(lines)
    all_velocities = np.hstack(velocities)

    # Create cell connectivity
    cells = []
    start_idx = 0
    for line in lines:
        n_points = len(line)
        cells.append(np.insert(np.arange(n_points) + start_idx, 0, n_points))
        start_idx += n_points
    cells = np.hstack(cells).astype(np.int64)

    # Create PyVista object
    poly = pv.PolyData()
    poly.points = all_points
    poly.lines = cells
    poly['velocity'] = all_velocities
    return poly


# Fluid surface
# surface = oil_iterator.get_surface(surface_idx)


pred_df = pd.read_csv(particle_pred_df_path)
track_pred = get_track(
    df=pred_df,
    x_key='x_pred',
    y_key='y_pred',
    z_key='z_pred',
    vx_key='vx_pred',
    vy_key='vy_pred',
    vz_key='vz_pred',
    drop_percent=drop_percent,
)

ground_df = pd.read_csv(particle_ground_df_path)
track_ground = get_track(
    df=ground_df,
    x_key='x',
    y_key='y',
    z_key='z',
    vx_key='vx',
    vy_key='vy',
    vz_key='vz',
    drop_percent=drop_percent,
)

p = pv.Plotter(
    title="Particle Prediction vs Ground Truth",
    shape=(1, 2),
    window_size=[2000, 1000])

# Window 1 - Ground Truth
p.show_grid(
    all_edges=True,
    show_xlabels=False,
    show_ylabels=False,
    show_zlabels=False,
)
p.add_text("Ground Truth", font_size=20)
p.add_mesh(
    track_ground,
    scalars='velocity',
    line_width=line_width,
    cmap=cmap,
    render_lines_as_tubes=True,
    opacity=opacity,
    clim=[np.quantile(track_ground['velocity'], clim_low),
          np.quantile(track_ground['velocity'], clim_high)],
)
# p.add_mesh(
#     surface,
#     color="blue",
#     pbr=True,
#     opacity=0.05)

# Window 2 - Prediction
p.subplot(0, 1)
p.show_grid(
    all_edges=True,
    show_xlabels=False,
    show_ylabels=False,
    show_zlabels=False,
)
p.add_text("Prediction", font_size=20)
p.add_mesh(
    track_pred,
    scalars='velocity',
    line_width=line_width,
    cmap=cmap,
    render_lines_as_tubes=True,
    opacity=opacity,
    clim=[np.quantile(track_ground['velocity'], clim_low),
          np.quantile(track_ground['velocity'], clim_high)],
)
# p.add_mesh(
#     surface,
#     color="blue",
#     pbr=True,
#     opacity=0.05)

p.link_views()

p.camera_position = "yz"
p.camera.azimuth = -30
p.camera.elevation = 15

if not save_fig:
    p.show()

elif save_fig:
    p.camera.azimuth = 0
    p.open_gif("compare_track_move_camera.gif", fps=4)
    p.camera.zoom(1.2)
    for i in range(num_frames):
        p.camera.azimuth = p.camera.azimuth - 1
        p.write_frame()
    p.close()
