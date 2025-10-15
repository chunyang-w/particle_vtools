"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Visulisation demo for Explorer3D class
"""

import glob
import numpy as np
from natsort import natsorted

from particle_vtools.Explorer3D import Explorer3D
from particle_vtools.PoreStructure import PoreStructure_CT
from particle_vtools.FluidStructure import FluidIterator_CT
from particle_vtools.Particle import ParticleIterator_DF

# Scaling factor - larger factor means smaller image
# a larger factore will accelerate the rendering

save_fig = False
clip_panel = True

down_sample_factor = 8
clim = [0, 10]
arrow_lim = [0.25, 3]

# # Change the paths to fit your data location
# Original data
pore_tif_path = "../data/Ketton155Data/141_ketton3_segmented_inverted.tif"  # noqa
# pore_tif_path = "../data/CombinedResults/Segmentations/074_segmented_tifs/seg_frame0.tif"  # noqa
ct_files_path = "../data/Segmentations/155_segmented_cleaned/*"  # noqa
particle_df_path = "../data/Velocity/155_Drain100x0p5s250nlmin_ft99p5_velocityPoints.csv"  # noqa

# Filtered data
# pore_tif_path = "../data/Ketton155Data/141_ketton3_segmented_inverted.tif"  # noqa
# ct_files_path = "/Users/chunyang/projects/particle/data/Ketton155Data/155_segmented_cleaned_filtered/*"  # noqa
# particle_df_path = "/Users/chunyang/projects/particle/data/Ketton155Data/155_Drain100x0p5s250nlmin_ft99p5_velocityPoints_surface_masked.csv"  # noqa

# Filtered croped
# pore_tif_path = "../data/Ketton155Data/141_ketton3_segmented_inverted.tif"  # noqa
# ct_files_path = "/Users/chunyang/projects/particle/data/Ketton155Data/155_segmented_cleaned_filtered_cropped/*"  # noqa
# particle_df_path = "/Users/chunyang/projects/particle/data/Ketton155Data/155_Drain100x0p5s250nlmin_ft99p5_velocityPoints_surface_masked.csv"  # noqa

if __name__ == "__main__":
    fluid_slicer = (slice(0, None), slice(0, None), slice(0, None))
    shift = np.array([100, 100, 100]).reshape(-1, 3)
    rock_surface = PoreStructure_CT(
        pore_tif_path,  # noqa
        scale=1,
        threshold=0,
        down_sample_factor=down_sample_factor,
        permute_axes=(2, 1, 0))

    # Load the oil surface
    ct_files = glob.glob(ct_files_path) # noqa
    ct_files = natsorted(ct_files)
    oil_iterator = FluidIterator_CT(
        "oil", ct_files, threshold=1,
        permute_axes=(2, 1, 0),
        down_sample_factor=down_sample_factor,
        slicer=fluid_slicer)
    # Particle data
    particle_df_path = particle_df_path # noqa
    paricle_iterator = ParticleIterator_DF(
        "particle",
        particle_df_path,
        shift_array=shift,
        arrow_lim=arrow_lim,
        scale_arrow=10,
        )

    # Init explorer
    explorer = Explorer3D(
        fluid_iterators=[oil_iterator],
        velocity_iterators=[paricle_iterator],
        # pore_structure=rock_surface,
        clim=clim,
        # clip_panel=clip_panel,
        clip_panel=True,
        num_frames=len(oil_iterator),
        surface_transparency=0.01,
        )


if not save_fig:
    explorer.set_scene3d(0)
    explorer.set_time_slider()
    p = explorer.plotter
    p.camera_position = "yz"
    p.camera.azimuth = 0
    p.camera.elevation = 15
    explorer.explore()
elif save_fig:
    move_camera = True
    num_frames = [i for i in range(45, 65, 1)]
    explorer.set_scene3d(num_frames[0])
    p = explorer.plotter
    p.camera_position = "yz"
    p.camera.azimuth = 0
    p.camera.elevation = 15
    p.camera.zoom(2)
    p.open_gif("compare_flow_direction.gif", fps=2)
    # p.open_movie("compare_flow_direction.mp4", framerate=2)

    text_actor = p.add_text(
        "Frame: 0", position="upper_right", font_size=20)

    for i in num_frames:
        p.write_frame()
        p.remove_actor(text_actor)
        text_actor = p.add_text(
            f"Frame: {i}", position="upper_right", font_size=20)
        explorer.update_scene3d(i)
        j = i / 10
        if move_camera:
            # p.camera.azimuth = p.camera.azimuth - j*2
            p.camera.elevation = p.camera.elevation + 1
    p.close()
