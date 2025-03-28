"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Visulisation script utilising pyvista API to compare the physics inside
a porous media.

A 3D scene with two subplots is created to compare the particle tracking
between the ground truth and the prediction.

Two view are linked to facilitate the comparison.
"""

# import glob
import pyvista as pv
import pandas as pd

# from natsort import natsorted

from particle_vtools.Explorer3D import Explorer3D
from particle_vtools.PoreStructure import PoreStructure_CT
from particle_vtools.FluidStructure import (  # noqa
    FluidIterator_CT, FluidIterator_Numpy
)
from particle_vtools.Particle import ParticleIterator_DF
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Particle Prediction vs Ground Truth Visualization")
parser.add_argument(
    "--save_fig", type=bool, default=False, help="Set to True to save the gif")
parser.add_argument(
    "--move_camera", type=bool, default=False, help="Set to True to move the camera for a better 3D view")  # noqa
parser.add_argument(
    "--num_frames", type=int, default=30, help="Number of frames to render")
parser.add_argument(
    "--show_clip_panel", type=bool, default=False, help="Set to True to show the clip panel")  # noqa
parser.add_argument(
    "--down_sample_factor", type=int, default=8, help="Scaling factor - larger factor means smaller image")  # noqa

args = parser.parse_args()

save_fig = args.save_fig
move_camera = args.move_camera
num_frames = args.num_frames
show_clip_panel = args.show_clip_panel
down_sample_factor = args.down_sample_factor

frame_start = 110
frame_end = 139
frame_offset = -1*frame_start
shift_array = [50, 50, -460]

# Change the paths to fit your data location
pore_tif_path = "/Users/chunyang/projects/particle/data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif"  # noqa

particle_pred_df_path = "/Users/chunyang/Downloads/072_autoregressive_noise5_predictions.csv"  # noqa
particle_ground_df_path = "/Users/chunyang/projects/particle/data/Velocity_smooth/072_final.csv"  # noqa

ct_numpy_path_pred = "/Users/chunyang/Downloads/pred (2).npy"
ct_numpy_path_gt = "/Users/chunyang/Downloads/cube_gt (2).npy"

if __name__ == "__main__":
    # fluid_slicer = (slice(None, -50), slice(50, -50), slice(50, -50))
    # shift = np.array([0, 0, -450]).reshape(-1, 3)
    scale = 8
    rock_surface = PoreStructure_CT(
        pore_tif_path,  # noqa
        scale=scale,
        threshold=0,
        down_sample_factor=down_sample_factor,
        permute_axes=(2, 1, 0))
    # rock_surface.tif_data = rock_surface.tif_data[230:, :, :]

    oil_iterator_gt = FluidIterator_Numpy(
        "oil_gt",
        ct_numpy_path_gt,
        threshold=1,
        scale=scale,
        permute_axes=(2, 1, 0),
        down_sample_factor=1,
        # slicer=fluid_slicer,
        frame_offset=frame_offset,
        )

    oil_iterator_pred = FluidIterator_Numpy(
        "oil_pred",
        ct_numpy_path_pred,
        threshold=1,
        scale=scale,
        permute_axes=(2, 1, 0),
        down_sample_factor=1,
        # slicer=fluid_slicer,
        frame_offset=frame_offset,
        )
    # Particle data
    pred_df = pd.read_csv(particle_pred_df_path)
    particle_idx = pred_df['particle'].unique()
    particle_iterator_pred = ParticleIterator_DF(
        "particle",
        particle_pred_df_path,
        frame_key='frame',
        x_key='x',
        y_key='y',
        z_key='z',
        vx_key='vx',
        vy_key='vy',
        vz_key='vz',
        frame_start=frame_start,
        frame_end=frame_end,
        scale_arrow=30,
        shift_array=shift_array,
        particle_idx=particle_idx,
        frame_offset=0,
    )

    particle_iterator_ground = ParticleIterator_DF(
        "particle",
        particle_ground_df_path,
        frame_key='frame',
        x_key='x',
        y_key='y',
        z_key='z',
        vx_key='vx',
        vy_key='vy',
        vz_key='vz',
        frame_start=frame_start,
        frame_end=frame_end,
        scale_arrow=30,
        shift_array=shift_array,
        particle_idx=particle_idx,
        frame_offset=0,
    )

    p = pv.Plotter(
        title="Particle Prediction vs Ground Truth",
        shape=(1, 2),
        window_size=[2000, 1000])
    # p.show_grid()

    # Init explorer
    explorer_pred = Explorer3D(
        fluid_iterators=[oil_iterator_pred],
        velocity_iterators=[particle_iterator_pred],
        pore_structure=rock_surface,
        num_frames=30,
        plotter=p,
        clip_panel=show_clip_panel,
        )

    explorer_ground = Explorer3D(
        fluid_iterators=[oil_iterator_gt],
        velocity_iterators=[particle_iterator_ground],
        pore_structure=rock_surface,
        num_frames=30,
        plotter=p,
        clip_panel=show_clip_panel,
        )

    def update_duo_view(frame_idx):
        p.subplot(0, 0)
        explorer_ground.update_scene3d(frame_idx)
        p.subplot(0, 1)
        explorer_pred.update_scene3d(frame_idx)

    p.subplot(0, 0)
    p.show_grid(
        all_edges=True,
        show_xlabels=False,
        show_ylabels=False,
        show_zlabels=False,
    )
    p.add_text("Ground Truth", font_size=20)

    explorer_ground.set_scene3d(frame_start)
    p.subplot(0, 1)
    p.show_grid(
        all_edges=True,
        show_xlabels=False,
        show_ylabels=False,
        show_zlabels=False,
    )

    p.add_text("Prediction", font_size=20)
    explorer_pred.set_scene3d(frame_start)
    p.link_views()

    p.camera_position = "yz"
    # Plus a positive azimuth is to rotate the camera clockwise
    p.camera.azimuth = 20
    p.camera.elevation = 15
    p.camera.zoom(1.8)

    if not save_fig:
        p.add_slider_widget(
            update_duo_view,
            [frame_start, frame_end],
            value=0,
            title='Frame',
        )
        p.show()

    elif save_fig:
        p.open_gif("73_pred_combine.gif", fps=8)
        text_actor = p.add_text(
            "Frame: 0", position="upper_right", font_size=20)
        for i in range(frame_start, frame_end):
            p.remove_actor(text_actor)
            text_actor = p.add_text(
                f"Frame: {i}", position="upper_right", font_size=20)
            update_duo_view(i)
            j = i / 10
            if move_camera:
                p.camera.azimuth = p.camera.azimuth - 1
            p.write_frame()
        p.close()
