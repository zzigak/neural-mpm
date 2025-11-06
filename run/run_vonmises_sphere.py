
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

    # Initialize sphere instead of loading from h5 file
    # Match H5 center position: (0.5, 0.5, 0.5) to keep sphere well within bounds
    mpm_solver.init_particles_sphere(
        center=(0.5, 0.5, 0.5),  # Center position matching H5 data (well within bounds)
        radius=0.05,              # Smaller radius to match H5 data span (~0.1 in x/y)
        n_grid=150,               # Grid resolution
        grid_lim=1.0,             # Grid domain size
        device=dvc
    )

    # Note: You must provide 'density=..' to set particle_mass = density * particle_volume

    material_params = {
        'E': 2000.0,
        'nu': 0.2,
        "material": "metal",  # Von Mises plasticity material
        'g': [0.0, 0.0, -4.0],
        "density": 200.0,
        "yield_stress": 100.0,  # Yield stress threshold for Von Mises plasticity
        "hardening": 1,          # Enable isotropic hardening (0 = no hardening, 1 = hardening)
        "xi": 0.1               # Hardening coefficient (how much yield stress increases with plastic strain)
    }
    mpm_solver.set_parameters(material_params)

    mpm_solver.compute_mu_lam_bulk() # set mu and lambda from the E and nu input

    mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', 0.0)

    directory_to_save = './results/vonmises_sphere'

    save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False)

    # Run simulation
    for k in range(1, num_frames + 1):
        mpm_solver.p2g2p(k, dt, device=dvc)
        save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)


if __name__ == "__main__":
    main()


