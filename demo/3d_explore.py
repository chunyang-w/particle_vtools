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

# Change the paths to fit your data location
pore_tif_path = "../data/073_combined_results/073_segmentedTimeSteps_downsampledx2_tif/073_segmented_00000.tif"  # noqa
ct_files_path = "../data/072_combined_results/registered_cleanedAvizo/*"  # noqa
particle_df_path = "../data/072_combined_results/trajectories/velocity_points_newversion_bp1_tsr6tm1tml20_surface_masked.csv"  # noqa

if __name__ == "__main__":
    fluid_slicer = (slice(None, -50), slice(50, -50), slice(50, -50))
    shift = np.array([0, 0, -450]).reshape(-1, 3)

    rock_surface = PoreStructure_CT(
        pore_tif_path,  # noqa
        scale=2,
        down_sample_factor=down_sample_factor,
        permute_axes=(2, 1, 0))
    rock_surface.tif_data = rock_surface.tif_data[230:, :, :]

    # Load the oil surface
    ct_files = glob.glob(ct_files_path) # noqa
    ct_files = natsorted(ct_files)
    oil_iterator = FluidIterator_CT(
        "oil", ct_files, 1,
        permute_axes=(2, 1, 0),
        down_sample_factor=down_sample_factor,
        slicer=fluid_slicer)
    # Particle data
    particle_df_path = particle_df_path# noqa
    paricle_iterator = ParticleIterator_DF("particle", particle_df_path, shift_array=shift)  # noqa

    # Init explorer
    explorer = Explorer3D([oil_iterator], [paricle_iterator], rock_surface)

    # set time slider
    explorer.set_time_slider()
    explorer.explore()
