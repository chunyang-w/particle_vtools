import pyvista as pv
import numpy as np
from natsort import natsorted
from skimage import io
from skimage import measure
import glob

save_fig = False
num_frames = 49
move_camera = False

dn_73 = natsorted(glob.glob("/Users/chunyang/projects/particle/data/Segmentations/073_downsampledx2/*.tif"))  # noqa
og_73 = natsorted(glob.glob("/Users/chunyang/projects/particle/data/Segmentations/073_segmented_tifs/*.tif")) # noqa

cube_og = io.imread(og_73[0])[::8, ::8, ::8]
cube_dn = io.imread(dn_73[0])[::4, ::4, ::4]


# Set up device
def cube2mesh(cube, threshold=0):
    # Extract the mesh from the cube
    verts, faces, _, _ = measure.marching_cubes(cube, level=threshold)
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    faces_pv = faces_pv.flatten()
    mesh = pv.PolyData(var_inp=verts, faces=faces_pv)
    mesh = mesh.smooth(n_iter=10, relaxation_factor=0.5)
    return mesh


# cube_pred = np.load("/Users/chunyang/Downloads/pred (1).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (2).npy")

t = 0
mesh_og = cube2mesh(cube_og, threshold=0)
mesh_dn = cube2mesh(cube_dn, threshold=0)

# Setup side-by-side plotting
plotter = pv.Plotter(shape=(1, 2), window_size=(1200, 600))

# Left plot: mesh_pred
plotter.subplot(0, 0)
plotter.add_mesh(mesh_og, color='lightblue')
plotter.add_text("Original Mesh", font_size=12)
plotter.add_axes()

# Right plot: mesh_gt
plotter.subplot(0, 1)
plotter.add_mesh(mesh_og, color='lightgreen')
plotter.add_text("Down sampled Mesh", font_size=12)
plotter.add_axes()

# Link camera views
plotter.link_views()


def update_slice(t):
    t = int(t)
    cube_pred = io.imread(og_73[t])[::8, ::8, ::8]
    cube_gt = io.imread(dn_73[t])[::8, ::8, ::8]
    mesh_pred_t1 = cube2mesh(cube_pred, threshold=0)
    mesh_gt_t1 = cube2mesh(cube_gt, threshold=0)
    mesh_og.points = mesh_pred_t1.points
    mesh_og.faces = mesh_pred_t1.faces
    mesh_dn.points = mesh_gt_t1.points
    mesh_dn.faces = mesh_gt_t1.faces


plotter.camera_position = "zx"
plotter.camera.azimuth = 80
plotter.camera.elevation = 15

if not save_fig:
    plotter.add_slider_widget(
        update_slice,
        [0, 200], value=0,
        title="Slice",
    )
    plotter.show()
elif save_fig:

    plotter.open_gif(f"compare_move_camera_{move_camera}.gif", fps=15)
    text_actor = plotter.add_text(
        "Frame: 0", position="upper_right", font_size=20)
    plotter.camera.zoom(1.2)
    for i in range(num_frames):
        plotter.remove_actor(text_actor)
        text_actor = plotter.add_text(
            f"Frame: {i}", position="upper_right", font_size=20)
        update_slice(i)
        if move_camera:
            plotter.camera.azimuth = plotter.camera.azimuth - 1
        plotter.write_frame()
    plotter.close()
