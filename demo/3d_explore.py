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

down_sample_factor = 8
clim = [0, 10]
arrow_lim = [0.5, 5]

# Change the paths to fit your data location
# pore_tif_path = "../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif"  # noqa
pore_tif_path = "../data/Segmentations/073_downsampledx2/073_segmented_00000.tif"
ct_files_path = "../data/Segmentations/072/*"  # noqa
particle_df_path = "../data/Velocity/072_velocity_points_newversion_bp1_tsr6tm1tml20_surface_masked.csv"  # noqa

if __name__ == "__main__":
    # The slicer is used to crop the fluid surface - this is optional
    fluid_slicer = (slice(None, -50), slice(50, -50), slice(50, -50))
    # shift array used to shift the particle data
    shift = np.array([0, 0, -450]).reshape(-1, 3)

    rock_surface = PoreStructure_CT(
        pore_tif_path,  # noqa
        scale=2,        # scale the image by two - the input image is too large this is optional  # noqa
        threshold=0,    # threshold used in marching cube algo to generate the surface  # noqa
        down_sample_factor=down_sample_factor*2, # down sample the image by 8 - this is optional  # noqa
        permute_axes=(2, 1, 0))  # permute the axes - this is optional  # noqa
    # delete this line if you do not want to crop the rock surface
    rock_surface.tif_data = rock_surface.tif_data[230:, :, :]

    # Load the oil surface
    ct_files = glob.glob(ct_files_path) # noqa
    ct_files = natsorted(ct_files)
    oil_iterator = FluidIterator_CT(
        "oil", ct_files, threshold=255,
        permute_axes=(2, 1, 0),
        down_sample_factor=down_sample_factor,
        slicer=fluid_slicer
        )

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
        )

    # set time slider
    explorer.set_scene3d(0)
    explorer.set_time_slider()
    explorer.explore()
