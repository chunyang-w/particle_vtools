"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

This file defines the entities that are used in the vtools package.

The entities are:
- ParticleIterator
"""
import pandas as pd
import numpy as np
import pyvista as pv

from abc import ABC, abstractmethod


class ParticleIterator(ABC):
    def __init__(self,
                 name,
                 frame_start=0,
                 ):
        self.name = name
        self.frame_start = frame_start

    @abstractmethod
    def get_particle(self, index):
        pass

    def get_glyph(self, index):
        """
        Abstract method that should return a mesh representing the pore
        surface.
        """
        positions, velocities = self.get_particle(index)
        points = pv.PolyData(positions)
        points['velocity'] = velocities
        arrow = pv.Arrow()
        glyphs = points.glyph(
            orient='velocity',
            color_mode='scale',
            factor=15,
            geom=arrow)
        return glyphs

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
        index = self.frame_start + index
        return self.get_glyph(index)

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


class ParticleIterator_DF(ParticleIterator):
    def __init__(self,
                 name,
                 df_path,
                 frame_key='frame',
                 x_key='x',
                 y_key='y',
                 z_key='z',
                 vx_key='vx',
                 vy_key='vy',
                 vz_key='vz',
                 shift_array=np.array([0, 0, 0]).reshape(-1, 3),
                 frame_start=0,
                 ):
        super().__init__(name, frame_start)
        self.df = pd.read_csv(df_path)
        self.frame_key = frame_key
        self.shift = shift_array
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.vx_key = vx_key
        self.vy_key = vy_key
        self.vz_key = vz_key

    def __len__(self):
        return self.df[self.frame_key].nunique()

    def get_frame(self, index):
        df_frame = self.df[self.df[self.frame_key] == index].copy()
        return df_frame

    def get_particle(self, index):
        # Read the TIFF file
        df_frame = self.get_frame(index)
        positions = df_frame[
            [self.x_key, self.y_key, self.z_key]].to_numpy() + self.shift
        velocities = df_frame[
            [self.vx_key, self.vy_key, self.vz_key]].to_numpy()

        positions = positions
        velocities = velocities

        return positions, velocities
