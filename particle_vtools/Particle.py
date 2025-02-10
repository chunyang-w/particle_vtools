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
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_particle(self, index):
        pass

    def get_glyph(self, index):
        """
        Abstract method that should return a mesh representing the pore
        surface.
        """
        positions, velocities, vel_mags = self.get_particle(index)
        points = pv.PolyData(positions)
        points['velocity'] = velocities
        points['velMags'] = vel_mags
        arrow = pv.Arrow()
        glyphs = points.glyph(
            orient='velocity',
            scale='velMags',
            color_mode='scale',
            factor=4,
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
                 shift_array=np.array([0, 0, 0]).reshape(-1, 3),
                 ):
        super().__init__(name)
        self.df = pd.read_csv(df_path)
        self.frame_key = frame_key
        self.shift = shift_array

    def __len__(self):
        return self.df[self.frame_key].nunique()

    def get_frame(self, index):
        df_frame = self.df[self.df[self.frame_key] == index].copy()
        return df_frame

    def get_particle(self, index):
        # Read the TIFF file
        df_frame = self.get_frame(index)
        positions = df_frame[['x', 'y', 'z']].to_numpy() + self.shift
        velocities = df_frame[['vx', 'vy', 'vz']].to_numpy()
        vel_mags = df_frame['velMags'].to_numpy()

        positions = positions
        velocities = velocities
        vel_mags = vel_mags

        return positions, velocities, vel_mags
