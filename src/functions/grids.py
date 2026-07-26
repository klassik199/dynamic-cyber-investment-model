import numpy as np


def build_grids(params: dict) -> dict:
    """
    Buduje siatki numeryczne dla zmiennych:
    t, lambda, H.
    """

    T = float(params["T"])
    lambda_max = float(params["lambda_max"])
    H_max = float(params["H_max"])

    N = int(params["N"])
    N_lambda = int(params["N_lambda"])
    N_h = int(params["N_h"])

    dt = T / N
    d_lambda = lambda_max / N_lambda
    dH = H_max / N_h

    t_grid = np.linspace(0.0, T, N + 1)
    lambda_grid = np.linspace(0.0, lambda_max, N_lambda + 1)
    H_grid = np.linspace(0.0, H_max, N_h + 1)

    return {
        "t": t_grid,
        "lambda": lambda_grid,
        "H": H_grid,
        "dt": dt,
        "d_lambda": d_lambda,
        "dH": dH,
    }