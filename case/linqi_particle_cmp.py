"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Visulisation script utilising pyvista API to compare the physics inside
a porous media.

A 3D scene with two subplots is created to compare the particle tracking
between the ground truth and the prediction.

Two view are linked to facilitate the comparison.
"""

import glob
import pyvista as pv

from natsort import natsorted
from particle_vtools.Explorer3D import Explorer3D
from particle_vtools.PoreStructure import PoreStructure_CT
from particle_vtools.FluidStructure import FluidIterator_CT
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
    "--frame_start", type=int, default=49, help="Number of frames to render")
parser.add_argument(
    "--frame_end", type=int, default=59, help="Number of frames to render")
parser.add_argument(
    "--show_clip_panel", type=bool, default=False, help="Set to True to show the clip panel")  # noqa
parser.add_argument(
    "--down_sample_factor", type=int, default=8, help="Scaling factor - larger factor means smaller image")  # noqa

args = parser.parse_args()

clim = [0, 0.5]

save_fig = args.save_fig
move_camera = args.move_camera
frame_start = args.frame_start
frames_end = args.frame_end
show_clip_panel = args.show_clip_panel
down_sample_factor = args.down_sample_factor


particle_pred_df_path = "/Users/chunyang/Downloads/test.csv"  # noqa
particle_ground_df_path = "/Users/chunyang/Downloads/test.csv"  # noqa

if __name__ == "__main__":
    # fluid_slicer = (slice(None, -50), slice(50, -50), slice(50, -50))
    # shift = np.array([0, 0, -450]).reshape(-1, 3)
    # scale = 2
    # rock_surface = PoreStructure_CT(
    #     pore_tif_path,  # noqa
    #     scale=scale,
    #     down_sample_factor=down_sample_factor,
    #     permute_axes=(2, 1, 0))
    # rock_surface.tif_data = rock_surface.tif_data[230:, :, :]

    # # Load the oil surface
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
    #     # slicer=fluid_slicer,
    #     )
    # Particle data
    particle_iterator_pred = ParticleIterator_DF(
        "particle",
        particle_pred_df_path,
        frame_key='frame',
        x_key='x_pred',
        y_key='y_pred',
        z_key='z_pred',
        vx_key='vx_pred',
        vy_key='vy_pred',
        vz_key='vz_pred',
        # frame_start=86,
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
        # frame_start=86,
    )

    p = pv.Plotter(
        title="Particle Prediction vs Ground Truth",
        shape=(1, 2),
        window_size=[2000, 1000])
    # p.show_grid()

    # Init explorer
    explorer_pred = Explorer3D(
        # fluid_iterators=[oil_iterator],
        velocity_iterators=[particle_iterator_pred],
        # pore_structure=rock_surface,
        num_frames=20,
        plotter=p,
        clip_panel=show_clip_panel,
        clim=clim,
        )

    explorer_ground = Explorer3D(
        # fluid_iterators=[oil_iterator],
        velocity_iterators=[particle_iterator_ground],
        # pore_structure=rock_surface,
        num_frames=20,
        plotter=p,
        clip_panel=show_clip_panel,
        clim=clim,
        )

    def update_duo_view(frame_idx):
        p.subplot(0, 0)
        # explorer_ground.update_scene3d(frame_idx)
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

    # explorer_ground.set_scene3d(frame_start)
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
    p.camera.azimuth = -30
    p.camera.elevation = 15

    if not save_fig:
        p.add_slider_widget(
            update_duo_view,
            [frame_start, frames_end],
            value=40,
            title='Frame',
        )
        p.show()

    elif save_fig:
        p.open_gif(f"compare_move_camera_{move_camera}.gif", fps=1.2)
        text_actor = p.add_text(
            f"Frame: {frame_start}", position="upper_right", font_size=20)
        p.camera.zoom(1.2)
        for i in range(frame_start, frames_end):
            p.remove_actor(text_actor)
            text_actor = p.add_text(
                f"Frame: {i}", position="upper_right", font_size=20)
            update_duo_view(i)
            j = i / 10
            if move_camera:
                p.camera.azimuth = p.camera.azimuth - 1*2
            p.write_frame()
        p.close()
