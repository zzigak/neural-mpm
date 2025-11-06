
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import warp as wp
from engine.solver import MPM_Solver
from engine.utils import *
import numpy as np


def main():
    wp.init()
    wp.config.verify_cuda = True

    dvc = "cuda:0"

    # Simulation parameters
    dt = 0.002  # time step
    num_frames = 600  # total number of frames to simulate

    mpm_solver = MPM_Solver(10) # initialize with whatever number is fine. it will be reintialized

    # Load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
    mpm_solver.load_from_sampling("data/sand_column.h5", n_grid = 150, device=dvc) 

    # Raise the object to drop from higher - shift all particles upward in z-direction
    drop_height = 0.3  # Raise object by this amount (adjust as needed)
    position_np = mpm_solver.mpm_state.particle_x.numpy()
    position_np[:, 2] = position_np[:, 2] + drop_height
    mpm_solver.mpm_state.particle_x = wp.from_numpy(position_np, dtype=wp.vec3, device=dvc)

    # Optionally, you can modify particle volumes after loading
    # Get volumes from numpy and modify if needed
    volume_np = mpm_solver.mpm_state.particle_vol.numpy()
    volume_np = np.ones(mpm_solver.n_particles) * 2.5e-8
    mpm_solver.mpm_state.particle_vol = wp.from_numpy(volume_np, dtype=float, device=dvc)

    # Note: You must provide 'density=..' to set particle_mass = density * particle_volume

    material_params = {
        'E': 2000.0,
        'nu': 0.2,
        "material": "jelly",  # elastic material
        'g': [0.0, 0.0, -4.0],
        "density": 200.0
    }
    mpm_solver.set_parameters_dict(material_params)

    mpm_solver.finalize_mu_lam_bulk() # set mu and lambda from the E and nu input

    mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', 0.0)

    directory_to_save = './sim_results/elastic'

    save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False)

    # Run simulation
    for k in range(1, num_frames + 1):
        mpm_solver.p2g2p(k, dt, device=dvc)
        save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)


if __name__ == "__main__":
    main()

