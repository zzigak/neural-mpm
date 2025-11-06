import warp as wp
from .utils import *
import numpy as np
import math


# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)

@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial

@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (2.0 * (1.0 + model.nu[p]))
    model.lam[p] = model.E[p] * model.nu[p]  / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))

@wp.kernel
def compute_bulk(state:MPMStateStruct, model:MPMModelStruct):
    p = wp.tid()
    model.bulk[p] = model.lam[p] + 2./3. * model.mu[p]

@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (
                        wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    ) * model.dx
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    C = state.particle_C[p]
                    # if model.rpic = 0, standard apic
                    C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                        C - wp.transpose(C)
                    )
                    if model.rpic_damping < -0.001:
                        # standard pic
                        C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    elastic_force = -state.particle_vol[p] * stress * dweight
                    v_in_add = (
                        weight
                        * state.particle_mass[p]
                        * (state.particle_v[p] + C * dpos)
                        + dt * elastic_force
                    )
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[grid_x, grid_y, grid_z]
        )
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = new_v
        state.particle_x[p] = state.particle_x[p] + dt * new_v
        state.particle_C[p] = new_C
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        # apply return mapping
        if model.material == 1:  # von Mises plasticity (metal)
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p
            )
        else:  # elastic (material == 0)
            state.particle_F[p] = state.particle_F_trial[p]

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wp.svd3(state.particle_F[p], U, sig, V)
        
        if model.material == 0:  # elastic
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        elif model.material == 1:  # von Mises plasticity
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )

        stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
        state.particle_stress[p] = stress


@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[grid_x, grid_y, grid_z] = (
        state.grid_v_out[grid_x, grid_y, grid_z] * scale
    )

