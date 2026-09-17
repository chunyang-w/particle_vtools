"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Visulisation script utilising pyvista API to compare the particle tracking

A 3D scene with two subplots is created to compare the particle tracking
between the ground truth and the prediction.

Two view are linked to facilitate the comparison.

Example Usage:

# visulise the 073 for validation (visulise)
python case/duo_track.py \
    --ct_files_path '../data/Segmentations/073_segmented_tifs/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/073.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/73_t150-180 (18).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[kalman]_073_[cross_mod]"

# 075
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/075_segmented_tifs/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/075.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/75_t66-96 (2).csv' \
    --frame_start 66 \
    --frame_end 96 \
    --tag "[long_multi][kalman]_075_[gns only]" \
    --crop_box 750 1100 850 1300 450 820 \
    --show_grid \

# 074
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/074_segmented_tifs/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/074.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/74_t150-180.csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[long_multi][kalman]_074_[multi-modal]" \
    --crop_box 750 1100 850 1300 450 820 \
    --show_grid \

python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t150-180 (20).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[long_multi][kalman]_155_[gns_only]" \
    --crop_box 750 1100 850 1300 450 820 \
    --show_grid \

python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t150-180 (18).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[fine_tune][kalman]_155_[gns_only]" \
    --crop_box 750 1100 850 1300 450 820 \
    --show_grid \

# Comment - this is not bad - slightly overestimate the velocity
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t150-180 (11).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[multiscale_cnn][kalman]_155_[gns_only]"

# Comment - not so good - overestimate the velocity
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t150-180 (12).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[global_multiscale_cnn][kalman]_155_[gns_only]"

# Comment - not bad acctually, slightly under estimate the overall velocity, and some spurious high velocity region.
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t150-180 (13).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[crop_GNS][kalman]_155_[gns_only]"

python case/duo_track.py \
    --ct_files_path '../data/Segmentations/073_segmented_tifs/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/073.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/73_t150-180 (17).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[kalman]_073_[cross_mod]"

python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t150-180 (10).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "[kalman]_155_[cross_mod]" \
    --crop_box 750 1100 250 900 450 820 \
    --show_grid \
    --crop_box 750 1100 850 1300 450 820 \
    --show_grid \

# Comment - this is not bad!
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t150-180 (9).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "0.2_noise_kalman" \
    --crop_box 750 1100 850 1300 450 820 \
    --show_grid


# Comment - not good - underestimate velocity
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (13).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "025_noise_randwalk_kalman"

# *Comment - this is not bad acctually
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (14).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "0.2_noise"

# Comment - this is not bad acctually, slightly under estimate the velocity for some reigon.
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (23).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "0.25_noise"

# Not bad, this is for 150-180 comparison.
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t150-180 (9).csv' \
    --frame_start 150 \
    --frame_end 180 \
    --tag "0.2_noise_kalman"

# Comment - far from good - underestimates the velocity
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (15).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "0.2_randwalk"

# Comment - error accumulating and cause sever swirling motion
python case/duo_track.py \
    --ct_files_path "../data/Segmentations/073_segmented_tifs/*" \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (24).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "0.05 randwalk noise"

# Comment - not very bad - under estimate the velocity a bit.
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (19).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "randwalk.25noise"

# Comment - not impressive - under estimate the velocity
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (22).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "randwalk.15noise"

# Comment - not very good - overestimate the velocity
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (16).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "0.1_noise_naive"

# Comment, acceptable - still overestimate the velocity
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (17).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "0.05_noise_naive"

# Promising, under estimate the velocity slightly
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (18).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "5_step_0.1noise"

# Comment - still slightly under estimate the velocity
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (20).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "5_step_0.05noise"

# Comment - still not very impressive - under estimate the velocity a bit.
python case/duo_track.py \
    --ct_files_path '/Users/chunyang/projects/particle/data/Segmentations/155_segmented_cleaned_filtered_cropped/*' \
    --particle_ground_df_path '/Users/chunyang/projects/particle/data/Velocity_kalman/155.csv' \
    --particle_pred_df_path '/Users/chunyang/Downloads/155_t110-140 (21).csv' \
    --frame_start 110 \
    --frame_end 140 \
    --tag "5_step_0.02noise"

"""

import glob
import numpy as np
import pandas as pd
import pyvista as pv

from natsort import natsorted
from particle_vtools.FluidStructure import FluidIterator_CT

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ct_files_path", type=str, required=True)
parser.add_argument("--particle_ground_df_path", type=str, required=True)
parser.add_argument("--particle_pred_df_path", type=str, required=True)
parser.add_argument("--frame_start", type=int, required=True)
parser.add_argument("--frame_end", type=int, required=True)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--crop_box", type=int, nargs=6, default=None)
parser.add_argument("--show_grid", action="store_true", default=False)
parser.add_argument("--save_fig", action="store_true", default=False)

args = parser.parse_args()

save_fig = args.save_fig

drop_percent = 0
clim_low = 0.2
clim_high = 0.95

down_sample_factor = 8
scale = 1

cmap = 'jet'
line_width = 5
opacity = 0.5

frame_start = args.frame_start
frame_end = args.frame_end

# particle_offset = [0, 0, -50]
particle_offset = [0, 0, 0]
show_labels = False

# particle_pred_df_path = "/Users/chunyang/Downloads/73_t125-155.csv"

surface_idx = -1


particle_max_height = 1180

show_bar = False

particle_ground_df_path = args.particle_ground_df_path
ct_files_path = args.ct_files_path
particle_pred_df_path = args.particle_pred_df_path

# Load the oil surface
ct_files = glob.glob(ct_files_path) # noqa
ct_files = natsorted(ct_files)
print(len(ct_files))
ct_files = ct_files[frame_start//2:frame_end//2]

print(ct_files)


def get_crop_df(df, crop_box):
    z_min, z_max, y_min, y_max, x_min, x_max = crop_box
    df_cropped = df[
        (df['z'] >= z_min) & (df['z'] < z_max) &
        (df['y'] >= y_min) & (df['y'] < y_max) &
        (df['x'] >= x_min) & (df['x'] < x_max)
    ]
    return df_cropped


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
    df = df[df[z_key] < particle_max_height]

    lines = []
    velocities = []
    particle_groups = df.groupby(particle_key)

    for pid, particle_data in particle_groups:
        # Extract ordered points for this particle
        points = particle_data[[x_key, y_key, z_key]].values
        points += particle_offset
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


pred_df = pd.read_csv(particle_pred_df_path)

if args.crop_box is not None:
    # The cropping box is defined as (z_min, z_max, y_min, y_max, x_min, x_max)
    crop_box = tuple(args.crop_box)
    fluid_slicer = (
        slice(args.crop_box[0], args.crop_box[1]),
        slice(args.crop_box[2], args.crop_box[3]),
        slice(args.crop_box[4], args.crop_box[5]),
        )
    pred_df = get_crop_df(pred_df, crop_box)
else:
    fluid_slicer = None


oil_iterator = FluidIterator_CT(
    "oil",
    ct_files,
    threshold=1,
    scale=scale,
    permute_axes=(2, 1, 0),
    down_sample_factor=down_sample_factor,
    slicer=fluid_slicer,
)

# Fluid surface
surface = oil_iterator.get_surface(surface_idx)
if args.crop_box is not None:
    z_min, z_max, y_min, y_max, x_min, x_max = crop_box
    offset = np.array([x_min, y_min, z_min])
    surface.points += offset


selected_idx = pred_df["particle"].unique()

track_pred = get_track(
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
    frame_end=frame_end,
    drop_percent=drop_percent,
    particle_offset=particle_offset,
    particle_idx=selected_idx,
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
    frame_key='frame',
    particle_key='particle',
    frame_start=frame_start,
    frame_end=frame_end,
    drop_percent=drop_percent,
    particle_offset=particle_offset,
    particle_idx=selected_idx,
)

p = pv.Plotter(
    title=f"Particle Prediction vs Ground Truth - {args.tag}",
    shape=(1, 2),
    window_size=[2500, 1250])

# Window 1 - Ground Truth
if args.show_grid:
    p.show_grid(
        all_edges=True,
        show_xlabels=show_labels,
        show_ylabels=show_labels,
        show_zlabels=show_labels,
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
    opacity=0.05)

# Window 2 - Prediction
p.subplot(0, 1)
if args.show_grid:
    p.show_grid(
        all_edges=True,
        show_xlabels=show_labels,
        show_ylabels=show_labels,
        show_zlabels=show_labels,
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
    opacity=0.05)

p.link_views()

p.camera_position = "yz"
p.camera.azimuth = -120
p.camera.elevation = 10
p.camera.zoom(1.2)

csv_base_name = particle_pred_df_path.split("/")[-1].split(".")[0]
run_name = f"out/duo_track_{csv_base_name}_{args.tag}"

if show_bar:
    pass
else:
    p.remove_scalar_bar()

if not save_fig:
    p.show()

elif save_fig:
    p.camera.azimuth = 0

    p.export_html(f"{run_name}.html")
    p.open_gif(f"{run_name}.gif", fps=4)
    num_frames = 100
    for i in range(num_frames):
        p.camera.azimuth = p.camera.azimuth - 1
        p.write_frame()
    p.close()
