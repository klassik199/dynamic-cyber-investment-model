import numpy as np


def build_grids(params: dict) -> dict:
    """Build uniform grids for time, attack intensity and security level."""
    T = float(params["T"])
    lambda_max = float(params["lambda_max"])
    H_max = float(params["H_max"])

    N = int(params["N"])
    N_lambda = int(params["N_lambda"])
    N_h = int(params["N_h"])

    dt = T / N
    d_lambda = lambda_max / N_lambda
    dH = H_max / N_h

    return {
        "t": np.linspace(0.0, T, N + 1),
        "lambda": np.linspace(0.0, lambda_max, N_lambda + 1),
        "H": np.linspace(0.0, H_max, N_h + 1),
        "dt": dt,
        "d_lambda": d_lambda,
        "dH": dH,
    }
