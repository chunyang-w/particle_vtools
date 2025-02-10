"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

This file defines the entities that are used in the vtools package.

The entities are:
- FluidIterator
"""
from abc import ABC, abstractmethod
from skimage import io
from .utils import tif_2_geo, geo_2_mesh


class FluidIterator(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_geo(self, index):
        """
        Abstract method that should return the geometry of the pore structure.
        This includes vertices and faces.
        """
        pass

    @abstractmethod
    def get_surface(self, index):
        """
        Abstract method that should return a mesh representing the pore
        surface.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Returns the total number of fluid files.
        """
        pass

    def __getitem__(self, index):
        """
        Supports indexing and slicing.
        """
        return self.get_surface(index)

    def __iter__(self):
        """
        Returns an iterator that iterates over velocity fields.
        """
        self._index = 0
        return self

    def __next__(self):
        """
        Returns the next velocity field in the iteration.
        """
        if self._index < len(self.fluid_files):
            result = self.get_velocity_field(self._index)
            self._index += 1
            return result
        else:
            raise StopIteration


class FluidIterator_CT(FluidIterator):
    def __init__(self,
                 name,
                 fluid_file_list,
                 threshold=1,
                 down_sample_factor=4,
                 smooth_iter=10,
                 smooth_factor=0.5,
                 scale=None,
                 permute_axes=None,
                 slicer=None):
        super().__init__(name)
        self.fluid_files = fluid_file_list
        self.threshold = threshold
        self.down_sample_factor = down_sample_factor
        self.smooth_iter = smooth_iter
        self.smooth_factor = smooth_factor
        self.scale = scale
        self.permute_axes = permute_axes
        self.slicer = slicer

    def get_geo(self, index):
        # Read the TIFF file
        tif_data = io.imread(self.fluid_files[index])
        if self.slicer:
            tif_data = tif_data[self.slicer]
        # Convert the TIFF data to geometry (vertices and faces)
        verts, faces = tif_2_geo(
            tif_data,
            threshold=self.threshold,
            down_sample_factor=self.down_sample_factor,
        )
        if self.scale:
            verts *= self.scale
        if self.permute_axes:
            verts = verts[:, self.permute_axes]
        return verts, faces

    def get_surface(self, index):
        verts, faces = self.get_geo(index)
        # Convert the geometry into a mesh
        mesh_surface = geo_2_mesh(
            verts, faces,
            smooth_iter=self.smooth_iter,
            smooth_factor=self.smooth_factor)
        return mesh_surface

    def __len__(self):
        return len(self.fluid_files)
