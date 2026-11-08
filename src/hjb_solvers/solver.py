"""Core routines for solving the HJB-PIDE by policy iteration on the lambda-H grid."""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve

from functions import (
    S_exp,
    S_rational,
    premium,
    investment_cost,
    terminal_utility,
)

def initialize_terminal_value(grids: dict) -> np.ndarray:
    """Terminal condition V(T,lambda,H) = -U(H) on the lambda-H grid."""

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    N_lambda_points = len(lambda_grid)

    terminal_row = -terminal_utility(H_grid)

    V_terminal = np.tile(terminal_row, (N_lambda_points, 1))

    return V_terminal

def initialize_strategy(grids: dict) -> np.ndarray:
    """Initial zero policy used as the first policy-iteration guess."""

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    z_initial = np.zeros((len(lambda_grid), len(H_grid)))

    return z_initial

def update_strategy(V: np.ndarray, grids: dict, params: dict, z_old: np.ndarray) -> np.ndarray:
    """Update the feedback control from the H-derivative of the value function."""

    dH = grids["dH"]
    H_grid = grids["H"]

    zeta = float(params["zeta"])
    gamma = float(params["gamma"])
    z_max = float(params["z_max"])
    rho = float(params["rho"])

    drift_H = z_old - rho * H_grid[None, :]

    D_H_V = derivative_H(V, dH, drift_H)

    z_new = (-D_H_V - zeta) / gamma
    z_new = np.maximum(z_new, 0.0)
    z_new = np.minimum(z_new, z_max)

    return z_new

def derivative_H(V: np.ndarray, dH: float, drift_H: np.ndarray) -> np.ndarray:
    """Approximate V_H with upwind differences depending on the sign of drift_H."""

    D = np.empty_like(V)

    D[:, 0] = (V[:, 1] - V[:, 0]) / dH
    D[:, -1] = (V[:, -1] - V[:, -2]) / dH

    D_forward = (V[:, 2:] - V[:, 1:-1]) / dH
    D_backward = (V[:, 1:-1] - V[:, :-2]) / dH
    
    drift_interior = drift_H[:, 1:-1]
    D[:, 1:-1] = np.where(drift_interior >= 0, D_forward, D_backward)

    return D
S
def derivative_lambda(V: np.ndarray, d_lambda: float, drift_lambda: np.ndarray) -> np.ndarray:
    """Approximate V_lambda with upwind differences depending on the sign of drift_lambda."""

    D = np.empty_like(V)

    D[0, :] = (V[1, :] - V[0, :]) / d_lambda
    D[-1, :] = (V[-1, :] - V[-2, :]) / d_lambda

    D_forward = (V[2:, :] - V[1:-1, :]) / d_lambda
    D_backward = (V[1:-1, :] - V[:-2, :]) / d_lambda
    
    drift_interior = drift_lambda[1:-1, None]
    D[1:-1, :] = np.where(drift_interior >= 0, D_forward, D_backward)

    return D

def hawkes_jump_interpolation(V: np.ndarray, grids: dict, params: dict) -> np.ndarray:
    """Interpolate V(lambda + beta, H) for the Hawkes jump term."""

    lambda_grid = grids["lambda"]
    d_lambda = grids["d_lambda"]
    beta = float(params["beta"])

    N_lambda = len(lambda_grid) - 1
    lambda_max = lambda_grid[-1]

    shifted_lambda = lambda_grid + beta
    shifted_lambda = np.minimum(shifted_lambda, lambda_max)

    j_left = np.floor(shifted_lambda / d_lambda).astype(int)
    j_left = np.clip(j_left, 0, N_lambda - 1)

    lambda_left = lambda_grid[j_left]
    theta = (shifted_lambda - lambda_left) / d_lambda

    I_beta_V = (1.0 - theta[:, None]) * V[j_left, :] + theta[:, None] * V[j_left + 1, :]

    return I_beta_V

def compute_running_terms(
    z: np.ndarray,
    grids: dict,
    params: dict,
    leakage_function: str = "exp",
) -> np.ndarray:
    """Compute investment cost, premium and expected retained loss on the full grid."""

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    v = float(params["v"])
    a_1 = float(params["a_1"])
    a_2 = float(params["a_2"])
    b = float(params["b"])

    pi_0 = float(params["pi_0"])
    r = float(params["r"])
    eta_d = float(params["eta_d"])

    zeta = float(params["zeta"])
    gamma = float(params["gamma"])

    if leakage_function == "exp":
        S_H = S_exp(H_grid, v, a_1)
    elif leakage_function == "rational":
        S_H = S_rational(H_grid, v, a_2, b)
    else:
        raise ValueError("leakage_function musi być równe 'exp' albo 'rational'.")

    pi_H = premium(S_H, v, pi_0, r)

    cost_z = investment_cost(z, zeta, gamma)

    lambda_matrix = lambda_grid[:, None]
    S_matrix = S_H[None, :]
    pi_matrix = pi_H[None, :]

    loss_term = eta_d * S_matrix * lambda_matrix

    running_terms = cost_z + pi_matrix + loss_term

    return running_terms

def state_index(j: int, k: int, N_h_points: int) -> int:
    """Convert a two-dimensional grid index to a flat index."""
    return j * N_h_points + k

def build_hjb_matrix(
    z_old: np.ndarray,
    grids: dict,
    params: dict,
) -> csr_matrix:
    """Build the sparse matrix for the linear HJB step under a fixed policy."""

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    dt = grids["dt"]
    d_lambda = grids["d_lambda"]
    dH = grids["dH"]

    alfa = float(params["alfa"])
    ksi = float(params["ksi"])
    rho = float(params["rho"])
    beta = float(params["beta"])

    N_lambda_points = len(lambda_grid)
    N_h_points = len(H_grid)

    size = N_lambda_points * N_h_points

    A = lil_matrix((size, size), dtype=float)

    lambda_max = lambda_grid[-1]

    shifted_lambda = lambda_grid + beta
    shifted_lambda = np.minimum(shifted_lambda, lambda_max)

    j_left_array = np.floor(shifted_lambda / d_lambda).astype(int)
    j_left_array = np.clip(j_left_array, 0, N_lambda_points - 2)

    lambda_left = lambda_grid[j_left_array]
    theta_array = (shifted_lambda - lambda_left) / d_lambda

    for j in range(N_lambda_points):
        lambda_j = lambda_grid[j]

        drift_lambda = ksi * (alfa - lambda_j)

        j_left = j_left_array[j]
        theta = theta_array[j]

        for k in range(N_h_points):
            h_k = H_grid[k]

            row = state_index(j, k, N_h_points)

            drift_H = z_old[j, k] - rho * h_k

            A[row, row] += 1.0 / dt

            if j == 0:
                col_right = state_index(j + 1, k, N_h_points)
                col_self = state_index(j, k, N_h_points)

                A[row, col_right] += drift_lambda / d_lambda
                A[row, col_self] += -drift_lambda / d_lambda

            elif j == N_lambda_points - 1:
                col_self = state_index(j, k, N_h_points)
                col_left = state_index(j - 1, k, N_h_points)

                A[row, col_self] += drift_lambda / d_lambda
                A[row, col_left] += -drift_lambda / d_lambda

            else:
                col_right = state_index(j + 1, k, N_h_points)
                col_left = state_index(j - 1, k, N_h_points)
                col_self = state_index(j, k, N_h_points)

                if drift_lambda >= 0:
                    A[row, col_right] += drift_lambda / d_lambda
                    A[row, col_self] += -drift_lambda / d_lambda
                else:
                    A[row, col_self] += drift_lambda / d_lambda
                    A[row, col_left] += -drift_lambda / d_lambda

            if k == 0:
                col_up = state_index(j, k + 1, N_h_points)
                col_self = state_index(j, k, N_h_points)

                A[row, col_up] += drift_H / dH
                A[row, col_self] += -drift_H / dH

            elif k == N_h_points - 1:
                col_self = state_index(j, k, N_h_points)
                col_down = state_index(j, k - 1, N_h_points)

                A[row, col_self] += drift_H / dH
                A[row, col_down] += -drift_H / dH

            else:
                col_up = state_index(j, k + 1, N_h_points)
                col_down = state_index(j, k - 1, N_h_points)
                col_self = state_index(j, k, N_h_points)

                if drift_H >= 0:
                    A[row, col_up] += drift_H / dH
                    A[row, col_self] += -drift_H / dH
                else:
                    A[row, col_self] += drift_H / dH
                    A[row, col_down] += -drift_H / dH

            col_self = state_index(j, k, N_h_points)
            col_jump_left = state_index(j_left, k, N_h_points)
            col_jump_right = state_index(j_left + 1, k, N_h_points)

            A[row, col_jump_left] += lambda_j * (1.0 - theta)
            A[row, col_jump_right] += lambda_j * theta
            A[row, col_self] += -lambda_j

    return A.tocsr()

def solve_hjb_linear_step(
    V_next: np.ndarray,
    z_old: np.ndarray,
    grids: dict,
    params: dict,
    leakage_function: str = "exp",
) -> np.ndarray:
    """Solve one implicit time step for a fixed policy."""

    dt = grids["dt"]

    A_old = build_hjb_matrix(z_old, grids, params)

    size = A_old.shape
    A = (2.0 / dt) * eye(size[0], format="csr") - A_old

    running_terms = compute_running_terms(
        z_old,
        grids,
        params,
        leakage_function=leakage_function,
    )

    rhs = V_next / dt + running_terms

    rhs_vector = rhs.ravel()

    V_new_vector = spsolve(A, rhs_vector)

    return V_new_vector.reshape(V_next.shape)
