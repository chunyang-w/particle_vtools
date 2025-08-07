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
from particle_vtools.PoreStructure import PoreStructure_CT
from particle_vtools.FluidStructure import FluidIterator_CT
# from particle_vtools.Particle import ParticleIterator_DF
# import argparse

save_fig = True
num_frames = 100
use_rock_surface = True

drop_percent = 0
clim_low = 0.2
clim_high = 0.95

down_sample_factor = 8
scale = 2
show_surface = True
surface_idx = 0

cmap = 'jet'
line_width = 10
opacity = 0.5

# ##############################                    Group 1                 ##############################
pore_tif_path = "/Users/chunyang/Downloads/lq/group_1_2images/pore_structure_group_1.tif"        # noqa
particle_pred_df_path = "/Users/chunyang/Downloads/lq/group_1_2images/particles_real_pred_data_relative_1.csv"  # noqa
particle_ground_df_path = "/Users/chunyang/Downloads/lq/group_1_2images/particles_real_pred_data_relative_1.csv"  # noqa
frame_start = 50
frame_end = 58


##############################                    Group 2                 ##############################
# pore_tif_path = "/Users/chunyang/Downloads/lq/group_2_2images/pore_structure_group_2.tif"        # noqa
# particle_pred_df_path = "/Users/chunyang/Downloads/lq/group_2_2images/particles_real_pred_data_relative_2.csv"  # noqa
# particle_ground_df_path = "/Users/chunyang/Downloads/lq/group_2_2images/particles_real_pred_data_relative_2.csv"  # noqa
# frame_start = 50
# frame_end = 58


# ##############################                    Group 3                 ##############################
# pore_tif_path = "/Users/chunyang/Downloads/lq/group_3_2images/pore_structure_group_3.tif"        # noqa
# particle_pred_df_path = "/Users/chunyang/Downloads/lq/group_3_2images/particles_real_pred_data_relative_3.csv"  # noqa
# particle_ground_df_path = "/Users/chunyang/Downloads/lq/group_3_2images/particles_real_pred_data_relative_3.csv"  # noqa
# frame_start = 50
# frame_end = 58

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

rock_strucutre = PoreStructure_CT(
    pore_tif_path,  # noqa
    scale=1,        # scale the image by two - the input image is too large this is optional  # noqa
    threshold=0,    # threshold used in marching cube algo to generate the surface  # noqa
    down_sample_factor=down_sample_factor*1, # down sample the image by 8 - this is optional  # noqa
    permute_axes=(2, 1, 0))  # permute the axes - this is optional  # noqa
rock_surface = rock_strucutre.get_surface()


def get_track(
        df,
        x_key='pred_x',
        y_key='pred_y',
        z_key='pred_z',
        vx_key='pred_vx',
        vy_key='pred_vy',
        vz_key='pred_vz',
        drop_percent=0.0,
        frame_start=0,
        frame_end=100,
        id_key='particle_id',
        select_ids=None,
        ):
    # Create trajectories for each particle
    df = df.sort_values([id_key, 'frame'])
    df = df[(df.frame >= frame_start) & (df.frame <= frame_end)]

    # Randomly select subset of particles
    keep_frac = (100 - drop_percent)/100
    if select_ids is None:
        unique_ids = df[id_key].unique()
        select_ids = np.random.choice(
                unique_ids,
                size=int(len(unique_ids)*keep_frac),
                replace=False)
    df = df[df[id_key].isin(select_ids)]

    lines = []
    velocities = []
    particle_groups = df.groupby(id_key)

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
select_ids = pred_df['particle_id'].unique()
track_pred = get_track(
    df=pred_df,
    x_key='pred_x',
    y_key='pred_y',
    z_key='pred_z',
    vx_key='pred_vx',
    vy_key='pred_vy',
    vz_key='pred_vz',
    id_key="particle_id",
    drop_percent=drop_percent,
    frame_start=frame_start,
    frame_end=frame_end,
    select_ids=select_ids,
)

ground_df = pd.read_csv(particle_ground_df_path)
track_ground = get_track(
    df=ground_df,
    x_key='real_x',
    y_key='real_y',
    z_key='real_z',
    vx_key='real_vx',
    vy_key='real_vy',
    vz_key='real_vz',
    id_key="particle_id",
    drop_percent=drop_percent,
    frame_start=frame_start,
    frame_end=frame_end,
    select_ids=select_ids,
)

p = pv.Plotter(
    title="Particle Prediction vs Ground Truth",
    shape=(1, 2),
    window_size=[2000, 1000])

# Window 1 - Ground Truth
p.subplot(0, 0)
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
    # clim=[np.quantile(track_ground['velocity'], clim_low),
    #       np.quantile(track_ground['velocity'], clim_high)],
)
if use_rock_surface:
    p.add_mesh(
        rock_surface,
        color="gray",
        pbr=True,
        opacity=0.1)

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
    # clim=[np.quantile(track_ground['velocity'], clim_low),
    #       np.quantile(track_ground['velocity'], clim_high)],
)
if use_rock_surface:
    p.add_mesh(
        rock_surface,
        color="gray",
        pbr=True,
        opacity=0.1)

p.link_views()

p.camera_position = "yz"
p.camera.azimuth = -30
p.camera.elevation = 15

file_name = pore_tif_path.split(".")[0]

if not save_fig:
    p.show()

elif save_fig:
    p.camera.azimuth = 0
    p.open_gif(f"{file_name}.gif", fps=15)
    p.camera.zoom(1.2)
    for i in range(num_frames):
        p.camera.azimuth = p.camera.azimuth - 1
        p.write_frame()
    p.close()
