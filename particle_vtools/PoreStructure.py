"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

This file defines the entities that are used in the vtools package.

The entities are:
- PoreStructure
- FluidIterator
- ParticleIterator
"""
from abc import ABC, abstractmethod
from skimage import io
from .utils import tif_2_geo, geo_2_mesh


class PoreStructure(ABC):
    """
    Abstract base class that represents the pore structure of a porous media.
    """
    @abstractmethod
    def get_geo(self):
        """
        Abstract method that should return the geometry of the pore structure.
        This includes vertices and faces.
        """
        pass

    @abstractmethod
    def get_surface(self):
        """
        Abstract method that should return a mesh representing the pore
        surface.
        """
        pass


class PoreStructure_CT(PoreStructure):
    """
    Concrete implementation of PoreStructure for CT scan data.
    First load the tif data, then convert it to a mesh.
    Apply smoothing to the mesh if necessary.
    """
    def __init__(self,
                 tif_file,
                 threshold=0,
                 down_sample_factor=4,
                 smooth_iter=10,
                 smooth_factor=0.5,
                 scale=None,
                 permute_axes=None,
                 slicer=None):
        self.tif_data = io.imread(tif_file)
        self.threshold = threshold
        self.down_sample_factor = down_sample_factor
        self.smooth_iter = smooth_iter
        self.smooth_factor = smooth_factor
        self.scale = scale
        self.permute_axes = permute_axes
        self.slicer = slicer

    def get_geo(self):
        # Convert the TIFF data to geometry (vertices and faces)
        if self.slicer:
            self.tif_data = self.tif_data[self.slicer]
        verts, faces = tif_2_geo(
            self.tif_data,
            threshold=self.threshold,
            down_sample_factor=self.down_sample_factor
        )
        if self.scale:
            verts *= self.scale
        if self.permute_axes:
            verts = verts[:, self.permute_axes]
        return verts, faces

    def get_surface(self):
        verts, faces = self.get_geo()
        # Convert the geometry into a mesh
        mesh_surface = geo_2_mesh(
            verts, faces,
            smooth_iter=self.smooth_iter)
        return mesh_surface
