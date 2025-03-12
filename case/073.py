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
arrow_lim = [0.5, 5]

# # Change the paths to fit your data location
pore_tif_path = "../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif"  # noqa
# pore_tif_path = "../data/CombinedResults/Segmentations/074_segmented_tifs/seg_frame0.tif"  # noqa
ct_files_path = "../data/Segmentations/073_segmented_tifs/*"  # noqa
particle_df_path = "../data/Velocity/073_RobuGlass3_drainage_174nl_min_run5_velocityPoints_surface_masked.csv"  # noqa

if __name__ == "__main__":
    fluid_slicer = (slice(0, None), slice(0, None), slice(0, None))
    shift = np.array([50, 50, 0]).reshape(-1, 3)

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
        "oil", ct_files, threshold=0,
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
        )

    # Init explorer
    explorer = Explorer3D(
        [oil_iterator],
        [paricle_iterator],
        rock_surface,
        clim=clim,
        clip_panel=clip_panel,
        num_frames=len(oil_iterator),
        )

    # set time slider
    explorer.plotter.show_grid(
        all_edges=True,
        # show_xlabels=False,
        # show_ylabels=False,
        # show_zlabels=False,
    )


if not save_fig:
    explorer.set_scene3d(0)
    explorer.set_time_slider()
    explorer.explore()
elif save_fig:
    move_camera = True
    num_frames = [i for i in range(45, 65, 1)]
    p = explorer.plotter
    explorer.set_scene3d(num_frames[0])
    p.camera_position = "yz"
    p.camera.azimuth = 130
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
