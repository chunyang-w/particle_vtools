"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Tool that utilise pyvista API to provide a interactive 3D scene
for users to explore the surface of flow in porous media.

2D slicers are also provided to allow users to explore the internal
structure of the porous media.

If information about velocity field is provided, the class will also
plot the velocity field within the surface of the porous media.

Users can interact with the 3D scene by rotating, zooming, and panning,
as well as using a slider bar to move along time axis.
"""
import pyvista as pv


class Explorer3D:
    def __init__(
        self,
        fluid_iterators=None,
        velocity_iterators=None,
        pore_structure=None,
        num_frames=100,
        bg_color="white",
        surface_transparency=0.05,
        particle_cmap="jet",
        plotter=None,
        clip_panel=True,
        clim=[0, 7],
    ):
        self.pore_structure = pore_structure
        self.fluid_iterators = fluid_iterators
        self.velocity_iterators = velocity_iterators
        self.num_frames = num_frames
        self.surface_transparency = surface_transparency
        self.particle_cmap = particle_cmap
        self.plotter = plotter
        self.clip_panel = clip_panel
        self.clim = clim

        self.setup(bg_color)
        self.set_light()

        self.fluid_surfaces = []
        self.velocity_arrows = []

    def setup(self, bg_color):
        if self.plotter is None:
            self.plotter = pv.Plotter(
                window_size=[1600, 1600],
                title="Particle-vtools (3D Explorer)")
            pv.global_theme.background = bg_color

    def set_light(self):
        light = pv.Light()
        light.set_direction_angle(30, 30)

    def set_scene3d(self, frame_idx):
        frame_idx = int(frame_idx)
        # set fulid surface
        if self.fluid_iterators is not None:
            for fluid_iterator in self.fluid_iterators:
                fluid_mesh = fluid_iterator[frame_idx]
                self.fluid_surfaces.append(fluid_mesh)
                self.plotter.add_mesh(
                    fluid_mesh,
                    color="blue",
                    pbr=True,
                    metallic=0.1,
                    roughness=0.01,
                    diffuse=1,
                    opacity=self.surface_transparency)
                if self.clip_panel:
                    self.plotter.add_mesh_clip_plane(
                        fluid_mesh,
                        normal='-z',
                        origin=fluid_mesh.center,
                        color="blue", outline_opacity=0.1)

        # set particle velocity arrow
        if self.velocity_iterators is not None:
            for velocity_iterator in self.velocity_iterators:
                velocity = velocity_iterator[frame_idx]
                actor = self.plotter.add_mesh(
                    velocity,
                    # cmap=self.particle_cmap,
                    cmap=self.particle_cmap,
                    clim=self.clim,
                    scalar_bar_args={
                        'title': "Velocity Magnitude"
                        },
                )
                self.velocity_arrows.append(actor)

        # set pore structure
        if self.pore_structure is not None:
            pore_mesh = self.pore_structure.get_surface()
            if self.clip_panel:
                self.plotter.add_mesh_clip_plane(
                    pore_mesh,
                    normal='x', origin=pore_mesh.center,
                    color="grey")

    def update_scene3d(self, frame_idx):
        frame_idx = int(frame_idx)
        print(f"Updating scene to frame {frame_idx}")
        if self.fluid_iterators is not None:
            for i, fluid_iterator in enumerate(self.fluid_iterators):
                fluid_surface_i = fluid_iterator[frame_idx]
                self.fluid_surfaces[i].points = fluid_surface_i.points
                self.fluid_surfaces[i].faces = fluid_surface_i.faces

        if self.velocity_iterators is not None:
            for i, velocity_iterator in enumerate(self.velocity_iterators):
                # Remove old arrows
                if self.velocity_arrows[i] is not None:
                    self.plotter.remove_actor(self.velocity_arrows[i])
                # Generate new velocity arrows
                velocity_arrow_i = velocity_iterator[frame_idx]
                # actor.SetVisibility(True)
                self.velocity_arrows[i] = self.plotter.add_mesh(
                    velocity_arrow_i,
                    cmap=self.particle_cmap,
                    clim=self.clim,
                    scalar_bar_args={
                        'title': "Velocity Magnitude"
                        },
                    )

    def set_time_slider(self, start=0):
        start = start
        end = start + self.num_frames - 1
        self.plotter.add_slider_widget(
            self.update_scene3d,
            [start, end],
            title='Frame', value=0)

    def auto_animation(self):
        self.set_scene3d(0)
        self.plotter.add_timer_event(
            max_steps=self.num_frames, duration=2300,
            callback=self.update_scene3d)

    def explore(self):
        self.plotter.show()
        pass
