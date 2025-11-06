import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import warp as wp
from engine.solver import MPM_Solver
from engine.utils import *
import numpy as np




def main():
    wp.init()
    wp.config.verify_cuda = True

    dvc = "cuda:0"
    dt = 0.002
    num_frames = 600

    mpm_solver = MPM_Solver(10)

    n_grid = 150
    grid_lim = 1.0
    dx = grid_lim / n_grid
    particle_spacing = dx / 2.0
    particle_volume = particle_spacing ** 3

    radius = 0.025
    center1 = np.array([0.5, 0.5, 0.45], dtype=np.float32)
    center2 = np.array([0.5, 0.5, 0.55], dtype=np.float32)

    def generate_sphere_positions(center, radius, particle_spacing):
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
        relative_pos = positions - center
        distances = np.linalg.norm(relative_pos, axis=1)
        mask = distances <= radius
        positions = positions[mask]
        return positions.astype(np.float32)

    positions1 = generate_sphere_positions(center1, radius, particle_spacing)
    positions2 = generate_sphere_positions(center2, radius, particle_spacing)
    all_positions = np.vstack([positions1, positions2])
    all_positions = np.ascontiguousarray(all_positions)

    print(f"Sphere 1 particles: {len(positions1)}")
    print(f"Sphere 2 particles: {len(positions2)}")
    print(f"Total particles: {len(all_positions)}")

    mpm_solver.dim, mpm_solver.n_particles = 3, len(all_positions)
    mpm_solver.initialize(mpm_solver.n_particles, n_grid, grid_lim, device=dvc)

    print("Particles initialized for two spheres.")

    mpm_solver.mpm_state.particle_x = wp.from_numpy(
        all_positions, dtype=wp.vec3, device=dvc
    )

    wp.launch(
        kernel=set_vec3_to_zero,
        dim=mpm_solver.n_particles,
        inputs=[mpm_solver.mpm_state.particle_v],
        device=dvc,
    )

    wp.launch(
        kernel=set_mat33_to_identity,
        dim=mpm_solver.n_particles,
        inputs=[mpm_solver.mpm_state.particle_F_trial],
        device=dvc,
    )

    particle_volume_array = np.ones(mpm_solver.n_particles, dtype=np.float64) * particle_volume
    particle_volume_array = np.squeeze(particle_volume_array, 0) if particle_volume_array.ndim > 1 else particle_volume_array
    mpm_solver.mpm_state.particle_vol = wp.from_numpy(
        particle_volume_array, dtype=float, device=dvc
    )

    print("Total particles: ", mpm_solver.n_particles)

    material_params = {
        'E': 2000.0,
        'nu': 0.2,
        "material": "jelly",
        'g': [0.0, 0.0, -4.0],
        "density": 200.0
    }
    mpm_solver.set_parameters_dict(material_params)
    mpm_solver.finalize_mu_lam_bulk()
    mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', 0.0)

    directory_to_save = './sim_results/elastic_two_spheres'
    save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False)
    for k in range(1, num_frames + 1):
        mpm_solver.p2g2p(k, dt, device=dvc)
        save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)


if __name__ == "__main__":
    main()


