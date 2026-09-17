import numpy as np
import pyvista as pv
import torch
import warnings
from skimage import measure


def get_bounding_box(
    domain_shape: tuple[int, int, int], box_size: int
) -> tuple[int, int, int, int, int, int]:
    """
    Generate a random bounding box inside a 3D domain.

    Parameters
    ----------
    domain_shape: (z, y, x)
        Shape of the full domain/voxel grid.
    box_size: int
        Size of the bounding box. If negative, returns full domain.
        If larger than any dimension, clamped to that dimension.

    Returns
    -------
    (z_min, z_max, y_min, y_max, x_min, x_max)
        Bounds are half-open: [min, max). Guaranteed to lie within domain.
    """
    if box_size < 0:
        return (0, domain_shape[0], 0, domain_shape[1], 0, domain_shape[2])

    result = []
    for dim_size in domain_shape:
        size = min(box_size, dim_size)
        max_start = dim_size - size
        start = np.random.randint(0, max_start + 1)
        result.extend([start, start + size])
    return tuple(result)


def crop_domain_voxel(voxel, bounding_box: tuple[int, int, int, int, int, int]):
    """
    Crop a voxel grid using a bounding box defined as (z_min, z_max, y_min, y_max, x_min, x_max).
    The bounds are clamped to the voxel dimensions to avoid out-of-range indexing.
    """
    # bounding_box = tuple(box_len // ds_factor for box_len in bounding_box)
    if len(bounding_box) != 6:
        raise ValueError("bounding_box must have 6 elements: (z_min, z_max, y_min, y_max, x_min, x_max)")
    z_min, z_max, y_min, y_max, x_min, x_max = bounding_box
    z_min = max(int(z_min), 0)
    y_min = max(int(y_min), 0)
    x_min = max(int(x_min), 0)

    z_max = min(int(z_max), voxel.shape[0])
    y_max = min(int(y_max), voxel.shape[1])
    x_max = min(int(x_max), voxel.shape[2])

    return voxel[z_min:z_max, y_min:y_max, x_min:x_max]


def crop_domain_graph(data, bounding_box: tuple[int, int, int, int, int, int], ds_factor: int):
    """
    Filter graph nodes and edges to those inside the bounding box.
    Data(x=[8319, 21], edge_index=[2, 49171], edge_attr=[49171, 7], y=[5, 8319, 3], particle_id=[8319], t=8, exp_id=72, image_3D=[8319, 4, 4, 4])

    The bounding box is ordered as (z_min, z_max, y_min, y_max, x_min, x_max).
    Node positions are assumed to be stored in the first three channels of `data.x`
    in (z, y, x) order. Target tensors (y, a), image patches (image_3D), particle_id,
    and batch indices (if present) are sliced accordingly. If no nodes fall inside
    the box, the original data is returned unchanged.
    """
    if len(bounding_box) != 6:
        raise ValueError("bounding_box must have 6 elements: (z_min, z_max, y_min, y_max, x_min, x_max)")

    if data.x is None or data.x.shape[1] < 3:
        return data

    z_min, z_max, y_min, y_max, x_min, x_max = bounding_box
    device = data.x.device

    positions = data.x[:, :3]
    node_mask = (
        (positions[:, 0] / ds_factor >= z_min)
        & (positions[:, 0] / ds_factor < z_max)
        & (positions[:, 1] / ds_factor >= y_min)
        & (positions[:, 1] / ds_factor < y_max)
        & (positions[:, 2] / ds_factor >= x_min)
        & (positions[:, 2] / ds_factor < x_max)
    )

    if node_mask.sum() < 2:
        warnings.warn("crop_domain_graph: less than 2 nodes fall inside the bounding box; returning original graph.")
        return data

    keep_idx = node_mask.nonzero(as_tuple=False).view(-1)

    # Remap node indices for edges
    idx_map = -torch.ones(data.num_nodes, device=device, dtype=torch.long)
    idx_map[keep_idx] = torch.arange(keep_idx.numel(), device=device, dtype=torch.long)

    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    new_edge_index = idx_map[data.edge_index[:, edge_mask]]

    new_data = data.clone()
    new_data.x = data.x[keep_idx]
    new_data.edge_index = new_edge_index

    if getattr(data, "edge_attr", None) is not None:
        new_data.edge_attr = data.edge_attr[edge_mask]

    if getattr(data, "y", None) is not None:
        new_data.y = data.y[:, keep_idx, :] if data.y.dim() == 3 else data.y[keep_idx]

    if getattr(data, "a", None) is not None:
        new_data.a = data.a[:, keep_idx, :] if data.a.dim() == 3 else data.a[keep_idx]

    if getattr(data, "image_3D", None) is not None:
        new_data.image_3D = data.image_3D[keep_idx]

    # if getattr(data, "batch", None) is not None:
    #     new_data.batch = data.batch[keep_idx]

    if hasattr(data, "particle_id"):
        pid = data.particle_id
        if isinstance(pid, torch.Tensor):
            new_data.particle_id = pid[keep_idx]
        elif isinstance(pid, np.ndarray):
            new_data.particle_id = pid[keep_idx.cpu().numpy()]
        else:
            new_data.particle_id = pid

    new_data.num_nodes = keep_idx.numel()
    return new_data


def tif_2_geo(tif_file, threshold=0, down_sample_factor=4):
    """
    Extract the geometry of the surface from a tif data.
    """
    # pad_width = 1
    img = tif_file == threshold
    # img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=1)
    img = img[::down_sample_factor, ::down_sample_factor, ::down_sample_factor]
    verts, faces, _, _ = measure.marching_cubes(img, level=0.5)
    # verts = verts - pad_width
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
