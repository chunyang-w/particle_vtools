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
import pyvista as pv
pv.global_theme.transparent_background = True

# Scaling factor - larger factor means smaller image
# a larger factore will accelerate the rendering

# num_frames = [i for i in range(150, 180, 1)]

num_frames = [i for i in range(45, 180, 1)]

save_fig = False
save_fig = True
clip_panel = True
clip_panel = False

down_sample_factor = 5
clim = [0, 10]
arrow_lim = [0.5, 2.5]

anim_frame = 100
split_gif = True
fps = 48

# # Change the paths to fit your data location
pore_tif_path = "../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif"  # noqa
# pore_tif_path = "../data/CombinedResults/Segmentations/074_segmented_tifs/seg_frame0.tif"  # noqa
ct_files_path = "../data/Segmentations/073_segmented_tifs/*"  # noqa
particle_df_path = "/Users/chunyang/projects/particle/data/Velocity_smooth/073_final.csv"  # noqa


def bezier_ease(t, points=np.array([[0, 0], [0.25, 0.1], [0.25, 1], [1, 1]])):
    """Returns a value between 0 and 1 based on Bezier interpolation."""
    t = np.clip(t, 0, 1)
    return (1 - t)**3 * points[0, 1] + 3 * (1 - t)**2 * t * points[1, 1] + \
           3 * (1 - t) * t**2 * points[2, 1] + t**3 * points[3, 1]


# Sigmoid easing (smooth middle)
def sigmoid_ease(t, k=5):
    """Sigmoid interpolation for smooth acceleration/deceleration."""
    return 1 / (1 + np.exp(-k * (t - 0.5)))


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
        scale_arrow=15,
        )

    # Init explorer
    explorer = Explorer3D(
        fluid_iterators=[oil_iterator],
        velocity_iterators=[paricle_iterator],
        pore_structure=rock_surface,
        clim=clim,
        clip_panel=clip_panel,
        # clip_panel=True,
        num_frames=len(oil_iterator),
        surface_transparency=0.01,
        show_colorbar=False,
        )

    p = explorer.plotter
    # p.show_bounds(location='all')
    explorer.set_scene3d(num_frames[0])
    clip_start = 0

    fluid_mesh = explorer.fluid_surfaces[0]
    pore_mesh = explorer.pore_mesh

    fluid_mesh, fluid_mesh_clip = fluid_mesh.clip(
        normal="x",
        origin=(0, 0, 0),
        return_clipped=True,
        value=0)
    pore_mesh, pore_mesh_clip = pore_mesh.clip(
        normal="x",
        origin=(0, 0, 0),
        return_clipped=True,
        value=0)

    p.add_mesh(fluid_mesh_clip, color="blue")
    p.add_mesh(pore_mesh_clip, color="grey")

    # init camera
    p.camera_position = "yz"
    p.camera.azimuth = 60
    p.camera.elevation = 15

    if not save_fig:
        explorer.set_time_slider(num_frames[0])
        explorer.explore()
    elif save_fig:
        move_camera = False
        p.camera.zoom(1)
        if not split_gif:
            p.open_gif("render/073_render_01.gif", fps=fps)
        text_actor = p.add_text(
            "",
            position="upper_right",
            font_size=20)
        point_arr = []
        face_arr = []

        if split_gif:
            p.open_gif("render/073_render_01.gif", fps=fps)
        # move pore
        for i in range(0, anim_frame):
            clip_range = 1500
            delta_x = clip_range / anim_frame
            progress = i / (anim_frame - 1)
            eased_progress = bezier_ease(progress)
            current_value = clip_range * (1 - eased_progress)
            p.write_frame()
            pore_mesh_clip.clip(
                normal="x",
                origin=(0, 0, 0),
                inplace=True,
                # value=clip_range - delta_x*i,
                value=current_value,
                )
            point_arr.append(pore_mesh_clip.points)
            face_arr.append(pore_mesh_clip.faces)

        num_rest_frames = 20
        for i in range(num_rest_frames):
            p.write_frame()

        if split_gif:
            p.open_gif("render/073_render_02.gif", fps=fps)
        # move fluid
        for i in range(0, anim_frame):
            clip_range = 800
            delta_x = clip_range / (anim_frame - 30)
            progress = i / (anim_frame - 1)
            eased_progress = sigmoid_ease(progress)  # Try sigmoid for contrast
            current_value = clip_range * (1 - eased_progress) - 200
            p.write_frame()
            # explorer.update_scene3d(i)
            fluid_mesh_clip.clip(
                inplace=True,
                # value=clip_range - delta_x*i,
                value=current_value,
            )

        num_rest_frames = 20
        for i in range(num_rest_frames):
            p.write_frame()

        if split_gif:
            p.open_gif("render/073_render_03.gif", fps=fps)
        # move pore back
        for i in range(0, anim_frame//2 - 10):
            p.write_frame()
            point = point_arr.pop()
            face = face_arr.pop()
            pore_mesh_clip.points = point
            pore_mesh_clip.faces = face

        if split_gif:
            p.open_gif("render/073_render_03.gif", fps=fps)

        # move camera
        slow_factor = 4
        num_moving_frames = slow_factor*len(num_frames)
        for i in range(num_moving_frames):
            move_angle = 12
            zoom_factor = 1.6
            elevation = 8
            delta_angle = move_angle / num_moving_frames
            delta_zoom = (zoom_factor - 1) / num_moving_frames
            delta_elevation = elevation / num_moving_frames
            p.write_frame()
            p.remove_actor(text_actor)
            text_actor = p.add_text(
                f"Frame: {num_frames[i//slow_factor]}", position="upper_right",
                font_size=20)
            p.camera.azimuth = p.camera.azimuth + delta_angle
            p.camera.zoom(1 + delta_zoom)
            p.camera.elevation = p.camera.elevation + delta_elevation
            if i % slow_factor == 0:
                explorer.update_scene3d(num_frames[i//slow_factor])

        num_rest_frames = 50
        for i in range(num_rest_frames):
            p.write_frame()

        p.close()
