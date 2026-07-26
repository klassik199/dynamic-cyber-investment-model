"""Run a sensitivity check for the quadratic investment cost parameter gamma."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grids import build_grids
from solver import (
    initialize_terminal_value,
    initialize_strategy,
    solve_policy_iteration_step,
)


def load_params(path: str = "model_params.json") -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)

    return params


def bilinear_interpolate(
    array: np.ndarray,
    lambda_value: float,
    H_value: float,
    grids: dict,
) -> float:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    d_lambda = grids["d_lambda"]
    dH = grids["dH"]

    N_lambda = len(lambda_grid) - 1
    N_h = len(H_grid) - 1

    lambda_value = min(max(lambda_value, lambda_grid[0]), lambda_grid[-1])
    H_value = min(max(H_value, H_grid[0]), H_grid[-1])

    j = int(np.floor((lambda_value - lambda_grid[0]) / d_lambda))
    k = int(np.floor((H_value - H_grid[0]) / dH))

    j = min(max(j, 0), N_lambda - 1)
    k = min(max(k, 0), N_h - 1)

    lambda_left = lambda_grid[j]
    H_left = H_grid[k]

    theta_lambda = (lambda_value - lambda_left) / d_lambda
    theta_H = (H_value - H_left) / dH

    value = (
        (1.0 - theta_lambda) * (1.0 - theta_H) * array[j, k]
        + theta_lambda * (1.0 - theta_H) * array[j + 1, k]
        + (1.0 - theta_lambda) * theta_H * array[j, k + 1]
        + theta_lambda * theta_H * array[j + 1, k + 1]
    )

    return float(value)


def solve_for_gamma(params_base: dict, gamma_value: float) -> dict:
    params = params_base.copy()


    params["N"] = 1000


    params["N_lambda"] = 50


    params["z_max"] = 100


    params["gamma"] = gamma_value


    params["H_max"] = 100


    params["N_h"] = 50

    rho = float(params["rho"])
    T = float(params["T"])
    H_0 = float(params["H_0"])
    z_max = float(params["z_max"])

    H_required = max(
        H_0,
        z_max / rho * (1.0 - np.exp(-rho * T)),
    )

    grids = build_grids(params)

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    print("\n" + "=" * 80)
    print(f"Test dla gamma = {gamma_value}")
    print("=" * 80)
    print(f"N = {params['N']}, N_lambda = {params['N_lambda']}, N_h = {params['N_h']}")
    print(f"H_required = {H_required:.6f}, H_max = {params['H_max']}, dH = {grids['dH']:.6f}")

    stable = True
    last_step = 0

    for step in range(params["N"]):
        V_current, z_current, iterations, diff_V, diff_z = solve_policy_iteration_step(
            V_next=V_next,
            z_start=z_start,
            grids=grids,
            params=params,
            leakage_function="exp",
            max_iter=20,
            verbose=False,
        )

        last_step = step + 1

        if step == 0 or (step + 1) % 250 == 0:
            t_current = params["T"] - (step + 1) * grids["dt"]
            share_z_max = (z_current >= params["z_max"] - 1e-8).mean()
            share_z_zero = (z_current <= 1e-8).mean()

            print(
                f"krok {step + 1:4d}/{params['N']}, "
                f"t = {t_current:.3f}, "
                f"iteracje = {iterations:2d}, "
                f"diff_V = {diff_V:.2e}, "
                f"diff_z = {diff_z:.2e}, "
                f"min V = {V_current.min():.4f}, "
                f"max V = {V_current.max():.4f}, "
                f"min z = {z_current.min():.4f}, "
                f"mean z = {z_current.mean():.4f}, "
                f"max z = {z_current.max():.4f}, "
                f"udział z=z_max = {share_z_max:.2%}, "
                f"udział z=0 = {share_z_zero:.2%}"
            )

        max_abs_V = max(abs(V_current.min()), abs(V_current.max()))

        if iterations == 20 or diff_V > 1e3 or diff_z > 1.0 or max_abs_V > 1e8:
            stable = False
            print("\nPrzerwano: brak zbieżności albo niestabilność numeryczna.")
            print(f"krok = {step + 1}")
            print(f"diff_V = {diff_V:.6e}")
            print(f"diff_z = {diff_z:.6e}")
            print(f"max_abs_V = {max_abs_V:.6e}")
            break

        V_next = V_current
        z_start = z_current

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    V_initial_state = bilinear_interpolate(V_next, lambda_0, H_0, grids)
    z_initial_state = bilinear_interpolate(z_start, lambda_0, H_0, grids)

    share_z_max_0 = (z_start >= params["z_max"] - 1e-8).mean()
    share_z_zero_0 = (z_start <= 1e-8).mean()

    zeta = float(params["zeta"])
    cost_at_zmax = zeta * z_max + 0.5 * gamma_value * z_max ** 2

    result = {
        "gamma": gamma_value,
        "z_max": z_max,
        "cost_at_zmax": float(cost_at_zmax),
        "H_required": float(H_required),
        "H_max": float(params["H_max"]),
        "N_h": int(params["N_h"]),
        "dH": float(grids["dH"]),
        "stable": stable,
        "last_step": last_step,
        "V_initial_state": float(V_initial_state),
        "z_initial_state": float(z_initial_state),
        "min_V_0": float(V_next.min()),
        "max_V_0": float(V_next.max()),
        "min_z_0": float(z_start.min()),
        "mean_z_0": float(z_start.mean()),
        "max_z_0": float(z_start.max()),
        "share_z_max_0": float(share_z_max_0),
        "share_z_zero_0": float(share_z_zero_0),
    }

    return result


def main() -> None:
    params_base = load_params()

    gamma_values = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

    results = []

    for gamma_value in gamma_values:
        result = solve_for_gamma(params_base, gamma_value)
        results.append(result)

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("Podsumowanie testu wrażliwości względem gamma")
    print("=" * 80)

    print(df.to_string(index=False))

    output_path = Path("gamma_sensitivity_results.csv")
    df.to_csv(output_path, index=False)

    print(f"\nZapisano wyniki do pliku: {output_path.resolve()}")


if __name__ == "__main__":
    main()
