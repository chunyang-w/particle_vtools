import numpy as np
import pyvista as pv

from skimage import measure


def tif_2_geo(tif_file, threshold=0, down_sample_factor=4):
    """
    Extract the geometry of the surface from a tif data.
    """
    img = tif_file
    img = img[::down_sample_factor, ::down_sample_factor, ::down_sample_factor]
    verts, faces, _, _ = measure.marching_cubes(img, level=threshold)
    verts = verts * down_sample_factor
    return verts, faces


def geo_2_mesh(verts, faces, smooth_iter=10, smooth_factor=0.5):
    """
    Convert the geometry to a pyvista mesh.
    then smooth the mesh - if necessary.
    """
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    faces_pv = faces_pv.flatten()
    mesh = pv.PolyData(var_inp=verts, faces=faces_pv)
    mesh = mesh.smooth(n_iter=10, relaxation_factor=0.5)
    return mesh
