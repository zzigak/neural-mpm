import sys
import os
import numpy as np

from .utils import *
from .kernels import *


class MPM_Solver:
    def __init__(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.re_init(n_particles, n_grid, grid_lim, device=device)
        # CFL control
        self.enable_cfl = True
        self.cfl_factor = 0.9  # Safety factor: use 90% of max allowed dt

    def re_init(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.n_particles = n_particles
        self._init_model(n_particles, n_grid, grid_lim, device)
        self._init_state(n_particles, n_grid, device)
        self._init_simulation_state()

    def _init_model(self, n_particles, n_grid, grid_lim, device):
        """Initialize the MPM model structure with grid and material parameters."""
        self.mpm_model = MPMModelStruct()
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim]
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )

        # Material property arrays
        self.mpm_model.E = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.nu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.mu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.lam = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.bulk = wp.zeros(shape=n_particles, dtype=float, device=device)

        # Material model selection: 0 is jelly (elastic), 1 is metal (von Mises)
        self.mpm_model.material = 0

        # Plasticity parameters
        self.mpm_model.yield_stress = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_model.hardening = 0
        self.mpm_model.xi = 0.0

        # Physics parameters
        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)
        self.mpm_model.rpic = 0.0  # 0.0 if no damping (apic). -1 if pic
        self.mpm_model.grid_v_damping_scale = 1.1  # globally applied

    def _init_state(self, n_particles, n_grid, device):
        """Initialize the MPM state structure with particle and grid arrays."""
        self.mpm_state = MPMStateStruct()

        # Particle arrays
        self.mpm_state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position
        self.mpm_state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity
        self.mpm_state.particle_F = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic
        self.mpm_state.particle_R = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle R rotation
        self.mpm_state.particle_F_trial = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # apply return mapping will yield
        self.mpm_state.particle_stress = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.mpm_state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.mpm_state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_C = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )

        # Grid arrays
        self.mpm_state.grid_m = wp.zeros(
            shape=(n_grid, n_grid, n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(n_grid, n_grid, n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(n_grid, n_grid, n_grid),
            dtype=wp.vec3,
            device=device,
        )

    def _init_simulation_state(self):
        """Initialize simulation state variables."""
        self.time = 0.0
        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

    # must give density. mass will be updated as density * volume
    def set_parameters(self, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            if kwargs["material"] == "jelly":
                self.mpm_model.material = 0  # elastic
            elif kwargs["material"] == "metal":
                self.mpm_model.material = 1  # von Mises plasticity
            else:
                raise TypeError("Undefined material type. Supported: 'jelly' (elastic), 'metal' (von Mises)")

        if "grid_lim" in kwargs:
            self.mpm_model.grid_lim = kwargs["grid_lim"]
        if "n_grid" in kwargs:
            self.mpm_model.n_grid = kwargs["n_grid"]
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        if "E" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.E, kwargs["E"]],
                device=device,
            )
        if "nu" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.nu, kwargs["nu"]],
                device=device,
            )
        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.yield_stress, val],
                device=device,
            )
        if "hardening" in kwargs:
            self.mpm_model.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            self.mpm_model.xi = kwargs["xi"]

        if "g" in kwargs:
            self.mpm_model.gravitational_accelaration = wp.vec3(kwargs["g"][0], kwargs["g"][1], kwargs["g"][2])

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )
        if "rpic" in kwargs:
            self.mpm_model.rpic = kwargs["rpic"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]


    def compute_mu_lam_bulk(self, device = "cuda:0"):
        wp.launch(kernel = compute_mu_lambda_from_E_nu, dim = self.n_particles, inputs = [self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel=compute_bulk, dim=self.n_particles, inputs=[self.mpm_state, self.mpm_model], device=device)

#---------------------------------- main simulation loop ----------------------------------
    def p2g2p(self, step, dt, device="cuda:0"):
        # Enforce CFL condition if enabled
        actual_dt = dt
        if getattr(self, "enable_cfl", False):
            try:
                particle_v = self.mpm_state.particle_v.numpy()
                vmax = float(np.max(np.abs(particle_v)))
                dx = float(self.mpm_model.dx)
                if step == 1:  # Print once at the start
                    print(f"[CFL] Enabled with factor={self.cfl_factor}, dx={dx:.6e}")
                
                if vmax > 1e-12:  # Avoid division by zero
                    max_allowed_dt = self.cfl_factor * dx / vmax
                    if max_allowed_dt < dt:
                        actual_dt = max_allowed_dt
                        if step % 10 == 0:  # Print more frequently
                            print(
                                f"[CFL] Step {step}: Reduced dt from {dt:.6e} to {actual_dt:.6e} "
                                f"(v_max={vmax:.4e}, dx={dx:.6e}, CFL factor={self.cfl_factor})"
                            )
                    elif step % 100 == 0:  # Print occasionally when CFL is satisfied
                        print(
                            f"[CFL] Step {step}: CFL satisfied (dt={dt:.6e}, v_max={vmax:.4e}, "
                            f"max_allowed={max_allowed_dt:.6e})"
                        )
                elif step % 100 == 0:  # Print when velocities are very small
                    print(f"[CFL] Step {step}: Velocities very small (v_max={vmax:.4e}), using dt={dt:.6e}")
            except Exception as e:
                if step == 1:
                    print(f"[CFL] Warning: CFL check failed: {e}")
                pass  # If check fails, use original dt
        
        grid_size = ( self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z,)

        wp.launch(
            kernel=reset_grid,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )

        # compute stress = stress(returnMap(F_trial))
        wp.launch(
            kernel=compute_stress_from_F_trial,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, actual_dt],
            device=device,
        )  # F and stress are updated

        # p2g
        wp.launch(
            kernel=p2g_apic_with_stress,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, actual_dt],
            device=device,
        )  # apply p2g'

        # grid update
        wp.launch(
            kernel=grid_normalization_and_gravity,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model, actual_dt],
            device=device,
        )

        if self.mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=apply_grid_damping,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale],
                device=device,
            )

        # apply BC on grid
        for k in range(len(self.grid_postprocess)):
            wp.launch(
                kernel=self.grid_postprocess[k],
                dim=grid_size,
                inputs=[
                    self.time,
                    actual_dt,
                    self.mpm_state,
                    self.mpm_model,
                    self.collider_params[k],
                ],
                device=device,
            )
            if self.modify_bc[k] is not None:
                self.modify_bc[k](self.time, actual_dt, self.collider_params[k])

        # g2p
        wp.launch(
            kernel=g2p,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, actual_dt],
            device=device,
        )  # x, v, C, F_trial are updated

        self.time = self.time + actual_dt

    # set particle densities to all_particle_densities, 
    def set_densities(self, all_particle_densities, device = "cuda:0"):
        all_particle_densities = all_particle_densities.clone().detach()
        self.mpm_state.particle_density = torch2warp_float(all_particle_densities, dvc=device)
        wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

#---------------------------------- boundary conditions ----------------------------------

    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "frictional":
            collider_param.surface_type = 2
        else:
            raise ValueError("surface must be 'sticky', 'slip', or 'frictional'")
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:  # sticky
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    if param.surface_type == 1:  # slip
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        v = v - normal_component * n  # Project out all normal component
                        state.grid_v_out[grid_x, grid_y, grid_z] = v
                    if param.surface_type == 2:  # frictional
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        v = v - wp.min(normal_component, 0.0) * n  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(v)  # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = v

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def add_bounding_box(self, start_time=0.0, end_time=999.0):
        """Add bounding box colliders on all 6 faces of the domain using surface colliders.
        This prevents particles from leaving the domain boundaries."""
        grid_lim = self.mpm_model.grid_lim
        
        # Add 6 surface colliders for the bounding box
        # Each face uses 'slip' surface type to allow sliding but prevent penetration
        self.add_surface_collider(
            point=(0.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            surface="slip",
            friction=0.0,
            start_time=start_time,
            end_time=end_time,
        )  # x = 0 face
        self.add_surface_collider(
            point=(grid_lim, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
            surface="slip",
            friction=0.0,
            start_time=start_time,
            end_time=end_time,
        )  # x = grid_lim face
        
        self.add_surface_collider(
            point=(0.0, 0.0, 0.0),
            normal=(0.0, 1.0, 0.0),
            surface="slip",
            friction=0.0,
            start_time=start_time,
            end_time=end_time,
        )  # y = 0 face
        self.add_surface_collider(
            point=(0.0, grid_lim, 0.0),
            normal=(0.0, -1.0, 0.0),
            surface="slip",
            friction=0.0,
            start_time=start_time,
            end_time=end_time,
        )  # y = grid_lim face
        
        self.add_surface_collider(
            point=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            surface="slip",
            friction=0.0,
            start_time=start_time,
            end_time=end_time,
        )  # z = 0 face
        self.add_surface_collider(
            point=(0.0, 0.0, grid_lim),
            normal=(0.0, 0.0, -1.0),
            surface="slip",
            friction=0.0,
            start_time=start_time,
            end_time=end_time,
        )  # z = grid_lim face

    # the h5 file should store particle initial position and volume.
    def load_from_h5(
        self, sampling_h5, n_grid=100, grid_lim=1.0, device="cuda:0"
    ):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(sampling_h5, "r")
        x, particle_volume = h5file["x"], h5file["particle_volume"]

        x = x[()].transpose()  # np vector of x # shape now is (n_particles, dim)

        self.dim, self.n_particles = x.shape[1], x.shape[0]

        self.re_init(self.n_particles, n_grid, grid_lim, device=device)

        print(
            "Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles"
        )
        particle_volume = np.squeeze(particle_volume, 0)

        self.mpm_state.particle_x = wp.from_numpy(
            x, dtype=wp.vec3, device=device
        )  # initialize warp array from np

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial deformation gradient is set to identity

        self.mpm_state.particle_vol = wp.from_numpy(
            particle_volume, dtype=float, device=device
        )

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)

    def init_particles_sphere(
        self,
        center=(0.0, 0.0, 0.0),
        radius=0.1,
        particle_spacing=None,
        particle_volume=None,
        n_grid=100,
        grid_lim=1.0,
        device="cuda:0",
    ):
        """
        Initialize a sphere object with particles.
        This function replicates the exact initialization procedure from load_from_sampling.
        
        Args:
            center: Center of the sphere (x, y, z)
            radius: Radius of the sphere
            particle_spacing: Spacing between particles (if None, uses dx/2)
            particle_volume: Volume per particle (if None, uses particle_spacing^3)
            n_grid: Grid resolution
            grid_lim: Grid domain size [0, grid_lim]^3
            device: Device to use
        """
        # Calculate dx first (like load_from_sampling doesn't need temp init)
        dx = grid_lim / n_grid
        
        if particle_spacing is None:
            particle_spacing = dx / 2.0
        
        if particle_volume is None:
            particle_volume = particle_spacing ** 3
        
        center = np.array(center, dtype=np.float32)
        
        # Ensure sphere fits within grid bounds [0, grid_lim]^3
        # Adjust center if needed to keep sphere well within bounds (like H5 data)
        min_bound = radius + 0.05  # Add small margin like H5 data
        max_bound = grid_lim - radius - 0.05
        center = np.clip(center, [min_bound, min_bound, min_bound], [max_bound, max_bound, max_bound])
        
        # Generate particles in a grid pattern within a bounding box
        # Keep sphere well within bounds (like H5 data which is in [0.45, 0.55] range)
        extent = radius
        x_start = center[0] - extent
        x_end = center[0] + extent
        y_start = center[1] - extent
        y_end = center[1] + extent
        z_start = center[2] - extent
        z_end = center[2] + extent
        
        x_range = np.arange(x_start, x_end, particle_spacing, dtype=np.float32)
        y_range = np.arange(y_start, y_end, particle_spacing, dtype=np.float32)
        z_range = np.arange(z_start, z_end, particle_spacing, dtype=np.float32)
        
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        # Filter particles inside sphere (distance from center <= radius)
        relative_pos = positions - center
        distances = np.linalg.norm(relative_pos, axis=1)
        mask = distances <= radius
        positions = positions[mask]
        
        # (positions is already (n, 3) shape, so this matches x[()].transpose() from H5)
        positions = positions.astype(np.float32)
        
        self.dim, self.n_particles = 3, len(positions)
        
        # Initialize with correct particle count (exactly like load_from_sampling - only once!)
        self.re_init(self.n_particles, n_grid, grid_lim, device=device)
        
        print("Particles initialized for sphere.")
        
        # Initialize particle positions exactly like load_from_sampling
        self.mpm_state.particle_x = wp.from_numpy(
            positions, dtype=wp.vec3, device=device
        )  # initialize warp array from np
        
        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero
        
        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial deformation gradient is set to identity
        
        # Set particle volumes exactly like load_from_sampling
        particle_volume_array = np.ones(self.n_particles, dtype=np.float64) * particle_volume
        particle_volume_array = np.squeeze(particle_volume_array, 0) if particle_volume_array.ndim > 1 else particle_volume_array
        self.mpm_state.particle_vol = wp.from_numpy(
            particle_volume_array, dtype=float, device=device
        )
        
        print("Total particles: ", self.n_particles)

