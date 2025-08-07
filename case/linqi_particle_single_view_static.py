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
    "--save_fig", type=bool, default=True, help="Set to True to save the gif")
parser.add_argument(
    "--move_camera", type=bool, default=True, help="Set to True to move the camera for a better 3D view")  # noqa
parser.add_argument(
    "--frame_start", type=int, default=40, help="Number of frames to render")
parser.add_argument(
    "--frame_end", type=int, default=50, help="Number of frames to render")
parser.add_argument(
    "--show_clip_panel", type=bool, default=False, help="Set to True to show the clip panel")  # noqa
parser.add_argument(
    "--down_sample_factor", type=int, default=8, help="Scaling factor - larger factor means smaller image")  # noqa

args = parser.parse_args()

clim = [0, 7]

save_fig = args.save_fig
move_camera = args.move_camera
frame_start = args.frame_start
frame_end = args.frame_end
show_clip_panel = args.show_clip_panel
down_sample_factor = args.down_sample_factor


particle_df_path = "/Users/chunyang/Downloads/lq/whole3images/sinteredGlass_unsmoothed_velocityPoints_smoothed.csv"  # noqa
pore_tif_path = "/Users/chunyang/Downloads/lq/whole3images/image_3d.tif"

if __name__ == "__main__":
    # Rock surface
    rock = PoreStructure_CT(
        pore_tif_path,  # noqa
        scale=0,        # scale the image by two - the input image is too large this is optional  # noqa
        threshold=0,    # threshold used in marching cube algo to generate the surface  # noqa
        down_sample_factor=down_sample_factor, # down sample the image by 8 - this is optional  # noqa
        permute_axes=(2, 1, 0))  # permute the axes - this is optional  # noqa
    rock_surface = rock.get_surface()

    # Particle data
    particle_iterator = ParticleIterator_DF(
        "particle",
        particle_df_path,
        frame_key='frame',
        x_key='x',
        y_key='y',
        z_key='z',
        vx_key='vx',
        vy_key='vy',
        vz_key='vz',
        frame_start=frame_start,
        scale_arrow=8,
        frame_end=frame_end,
    )

    print("len(particle_iterator_pred):", len(particle_iterator))

    p = pv.Plotter(
        title="Particle Prediction vs Ground Truth",
        shape=(1, 1),
        window_size=[1000, 1400])

    # Init explorer
    explorer = Explorer3D(
        # fluid_iterators=[oil_iterator],
        velocity_iterators=[particle_iterator],
        # pore_structure=rock_surface,
        num_frames=frame_end - frame_start,
        plotter=p,
        clip_panel=show_clip_panel,
        clim=clim,
        # show_colorbar=False,
        )

    def update_view(frame_idx):
        explorer.update_scene3d(frame_idx)

    explorer.set_scene3d(frame_start)

    # p.add_mesh(
    #     rock_surface,
    #     color="gray",
    #     pbr=True,
    #     opacity=0.05)
    explorer.set_scene3d(frame_start)
    # p.remove_scalar_bar()

    p.camera_position = "yz"
    p.camera.azimuth = -30
    p.camera.elevation = 15

    if not save_fig:
        p.add_slider_widget(
            update_view,
            [frame_start, frame_end],
            value=frame_start,
            title='Frame',
        )
        p.show()

    elif save_fig:
        update_view(frame_start)
        file_name = particle_df_path.split(".")[0]+f"Arrow_frame_{frame_start}_no_rock.gif"
        print(file_name)
        p.open_gif(file_name, fps=24)
        p.camera.zoom(1.2)
        for i in range(100):
            if move_camera:
                p.camera.azimuth = p.camera.azimuth - 1
            p.write_frame()
        p.close()
