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
import pandas as pd
import numpy as np
from natsort import natsorted
from skimage import measure
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
    "--move_camera", type=bool, default=False, help="Set to True to move the camera for a better 3D view")  # noqa
parser.add_argument(
    "--num_frames", type=int, default=20, help="Number of frames to render")
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

frame_start = 150
frame_end = 180
shift_array = [50, 50, 0]

# Change the paths to fit your data location
pore_tif_path = "/Users/chunyang/projects/particle/data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif"  # noqa
particle_pred_df_path = "/Users/chunyang/Downloads/073_autoregressive_5noise_predictions.csv"  # noqa
particle_ground_df_path = "/Users/chunyang/projects/particle/data/Velocity_smooth/073_final.csv"  # noqa
ct_files_path = "/Users/chunyang/projects/particle/data/Segmentations/073_downsampledx2/*"  # noqa

cube_pd = np.load("/Users/chunyang/Downloads/73_t150-180_pd.npy")
cube_gt = np.load("/Users/chunyang/Downloads/73_t150-180_gt.npy")


def cube2mesh(
    cube,
    threshold=1,
    smooth=True,
    smooth_iter=10,
    relaxation_factor=0.1,
):  # noqa E501
    """
    Extract the geometry of the surface from a tif data.
    as well as surface smoothing.
    """
    cube = cube == threshold
    cube = cube.transpose(2, 1, 0)
    verts, faces, _, _ = measure.marching_cubes(cube, level=0)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(
        np.int64
    )  # noqa E501
    faces_pv = faces_pv.flatten()
    mesh = pv.PolyData(var_inp=verts, faces=faces_pv)
    if smooth:
        mesh = mesh.smooth(
            n_iter=smooth_iter, relaxation_factor=relaxation_factor
        )  # noqa E501
    mesh.face_raw = faces
    return mesh


def plot_mesh(t, cube, mesh, scale=8):
    t = int(t)
    mesh_new = cube2mesh(
        cube[t],
        threshold=0,
        relaxation_factor=0.2,
        smooth_iter=20,
    )
    mesh.points = mesh_new.points*scale
    mesh.faces = mesh_new.faces
    # print("showing frame t: ", t)
    return mesh_new


if __name__ == "__main__":
    # fluid_slicer = (slice(None, -50), slice(50, -50), slice(50, -50))
    # shift = np.array([0, 0, -450]).reshape(-1, 3)
    scale = 8
    rock_surface = PoreStructure_CT(
        pore_tif_path,  # noqa
        scale=2,
        threshold=0,
        down_sample_factor=down_sample_factor,
        permute_axes=(2, 1, 0))
    # rock_surface.tif_data = rock_surface.tif_data[230:, :, :]

    # Load the oil surface
    ct_files = glob.glob(ct_files_path) # noqa
    ct_files = natsorted(ct_files)

    oil_iterator = FluidIterator_CT(
        "oil",
        ct_files,
        threshold=2,
        scale=scale,
        permute_axes=(2, 1, 0),
        down_sample_factor=down_sample_factor,
        # slicer=fluid_slicer,
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
        scale_arrow=15,
        shift_array=shift_array,
        particle_idx=particle_idx,
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
        scale_arrow=15,
        arrow_lim=(0.3, 3.2),
        shift_array=shift_array,
        particle_idx=particle_idx,
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
        pore_structure=rock_surface,
        num_frames=30,
        plotter=p,
        clip_panel=show_clip_panel,
        surface_transparency=0.07,
        )

    explorer_ground = Explorer3D(
        # fluid_iterators=[oil_iterator],
        velocity_iterators=[particle_iterator_ground],
        pore_structure=rock_surface,
        num_frames=30,
        plotter=p,
        clip_panel=show_clip_panel,
        surface_transparency=0.07,
        )

    def update_duo_view(frame_idx):
        idx_from_zero = frame_idx - frame_start
        p.subplot(0, 0)
        explorer_ground.update_scene3d(frame_idx)
        plot_mesh(idx_from_zero, cube_gt, mesh_gt)
        p.subplot(0, 1)
        explorer_pred.update_scene3d(frame_idx)
        plot_mesh(idx_from_zero, cube_pd, mesh_pd)

    p.subplot(0, 0)
    # p.show_grid(
    #     all_edges=True,
    #     show_xlabels=False,
    #     show_ylabels=False,
    #     show_zlabels=False,
    # )
    p.add_text("Ground Truth", font_size=20)

    explorer_ground.set_scene3d(frame_start)

    mesh_gt = cube2mesh(cube_gt[0], threshold=0)
    mesh_gt.points = mesh_gt.points*scale
    mesh_gt.smooth(n_iter=10, relaxation_factor=0.5)
    p.add_mesh(mesh_gt, color='blue', opacity=0.02, style='wireframe', line_width=0.5)

    p.subplot(0, 1)
    # p.show_grid(
    #     all_edges=True,
    #     show_xlabels=False,
    #     show_ylabels=False,
    #     show_zlabels=False,
    # )

    p.add_text("Prediction", font_size=20)
    explorer_pred.set_scene3d(frame_start)

    mesh_pd = cube2mesh(cube_pd[0], threshold=0)
    mesh_pd.points = mesh_pd.points*scale
    mesh_pd.smooth(n_iter=10, relaxation_factor=0.5)
    p.add_mesh(mesh_pd, color='blue', opacity=0.02, style='wireframe', line_width=0.5)
    p.link_views()

    p.camera_position = "yz"
    p.camera.azimuth = -145
    p.camera.elevation = -5
    p.camera.zoom(1.6)

    if not save_fig:
        p.add_slider_widget(
            update_duo_view,
            [frame_start, frame_end],
            value=0,
            title='Frame',
        )
        p.show()

    elif save_fig:
        p.open_movie(f"compare_move_camera_{move_camera}.mp4", framerate=8)
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
