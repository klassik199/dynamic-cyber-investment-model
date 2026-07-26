"""Solve the Poisson benchmark with a constant intensity matched to the expected Hawkes attack count."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_PARAMS_PATH = Path("model_params.json")
DEFAULT_OUTPUT_ROOT = Path(".")

REQUIRED_PARAMS = (
    "a_1", "a_2", "b", "v",
    "lambda_0", "alfa", "ksi", "beta",
    "H_0", "rho", "H_max",
    "pi_0", "eta_d", "r",
    "zeta", "gamma", "z_max",
    "T", "N", "N_h", "eps_z",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rozwiązuje pełną politykę optymalną dla modelu Poissona "
            "o stałej intensywności dobranej do E[N_T] procesu Hawkesa."
        )
    )
    parser.add_argument(
        "--variant",
        choices=("S1", "S2", "both"),
        default="both",
        help="Funkcja prawdopodobieństwa wycieku (domyślnie: both).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="Ścieżka do pliku model_params.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Folder nadrzędny wyników.",
    )
    parser.add_argument(
        "--poisson-lambda",
        type=float,
        default=None,
        help=(
            "Opcjonalna stała intensywność Poissona. "
            "Domyślnie lambda_P = E_Hawkes[N_T]/T."
        ),
    )
    parser.add_argument(
        "--max-policy-iter",
        type=int,
        default=100,
        help="Maksymalna liczba iteracji polityki w kroku czasowym.",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.5,
        help="Współczynnik tłumienia aktualizacji polityki z przedziału (0,1].",
    )
    parser.add_argument(
        "--save-value-policy",
        action="store_true",
        help="Zapisuje także pełną tablicę funkcji wartości V(t,H).",
    )
    return parser.parse_args()

def load_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku parametrów: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        params = json.load(file)

    missing = [name for name in REQUIRED_PARAMS if name not in params]
    if missing:
        raise KeyError(
            "W pliku parametrów brakuje pól: " + ", ".join(missing)
        )

    return params

def expected_hawkes_count(params: dict) -> float:
    """Expected Hawkes attack count over the horizon [0,T]."""
    alpha = float(params["alfa"])
    xi = float(params["ksi"])
    beta = float(params["beta"])
    lambda0 = float(params["lambda_0"])
    T = float(params["T"])

    k = xi - beta
    if k <= 0:
        raise ValueError(
            "Warunek stabilności nie jest spełniony: ksi-beta musi być dodatnie."
        )

    lambda_inf = xi * alpha / k

    return (
        lambda_inf * T
        + (lambda0 - lambda_inf) * (1.0 - math.exp(-k * T)) / k
    )

def leakage_probability(H: np.ndarray, params: dict, variant: str) -> np.ndarray:
    v = float(params["v"])

    if variant == "S1":
        return v * np.exp(-float(params["a_1"]) * H)

    if variant == "S2":
        return v / (
            1.0 + float(params["a_2"]) * H
        ) ** float(params["b"])

    raise ValueError(f"Nieobsługiwany wariant: {variant}")

def premium(H: np.ndarray, S: np.ndarray, params: dict) -> np.ndarray:
    pi0 = float(params["pi_0"])
    r = float(params["r"])
    v = float(params["v"])

    return pi0 * (1.0 - r * (1.0 - S / v))

def utility(H: np.ndarray) -> np.ndarray:
    """Terminal utility U(H)=sqrt(H)."""
    return np.sqrt(np.maximum(H, 0.0))

def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Thomas algorithm for tridiagonal linear systems."""
    n = diagonal.size

    if rhs.size != n:
        raise ValueError("Niezgodny rozmiar prawej strony układu.")

    b = diagonal.astype(np.float64, copy=True)
    c = upper.astype(np.float64, copy=True)
    d = rhs.astype(np.float64, copy=True)

    eps = 1e-14

    for i in range(1, n):
        pivot = b[i - 1]
        if abs(pivot) < eps:
            raise np.linalg.LinAlgError(
                f"Zerowy pivot w algorytmie Thomasa, indeks {i-1}."
            )

        multiplier = lower[i - 1] / pivot
        b[i] -= multiplier * c[i - 1]
        d[i] -= multiplier * d[i - 1]

    if abs(b[-1]) < eps:
        raise np.linalg.LinAlgError("Zerowy końcowy pivot w algorytmie Thomasa.")

    solution = np.empty(n, dtype=np.float64)
    solution[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        if abs(b[i]) < eps:
            raise np.linalg.LinAlgError(
                f"Zerowy pivot podczas substytucji wstecznej, indeks {i}."
            )
        solution[i] = (d[i] - c[i] * solution[i + 1]) / b[i]

    return solution

def spatial_derivative(V: np.ndarray, dH: float) -> np.ndarray:
    """Approximate V_H with centered differences and one-sided boundary differences."""
    derivative = np.empty_like(V, dtype=np.float64)

    derivative[1:-1] = (V[2:] - V[:-2]) / (2.0 * dH)
    derivative[0] = (V[1] - V[0]) / dH
    derivative[-1] = (V[-1] - V[-2]) / dH

    return derivative

def solve_fixed_policy_step(
    V_next: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    dH: float,
    dt: float,
    S: np.ndarray,
    premium_values: np.ndarray,
    lambda_poisson: float,
    params: dict,
) -> np.ndarray:
    """Solve one implicit Poisson-model time step for a fixed policy."""
    rho = float(params["rho"])
    zeta = float(params["zeta"])
    gamma = float(params["gamma"])
    eta_d = float(params["eta_d"])

    drift = z - rho * H

    rate_up = np.maximum(drift, 0.0) / dH
    rate_down = np.maximum(-drift, 0.0) / dH

    rate_down[0] = 0.0
    rate_up[-1] = 0.0

    diagonal = 1.0 + dt * (rate_up + rate_down)
    upper = -dt * rate_up[:-1]
    lower = -dt * rate_down[1:]

    investment_cost = zeta * z + 0.5 * gamma * z**2
    retained_loss = eta_d * S * lambda_poisson
    running_cost = investment_cost + premium_values + retained_loss

    rhs = V_next + dt * running_cost

    return solve_tridiagonal(lower, diagonal, upper, rhs)

def solve_poisson_policy(
    params: dict,
    variant: str,
    lambda_poisson: float,
    max_policy_iter: int,
    damping: float,
    save_value_policy: bool,
) -> dict:
    if not (0.0 < damping <= 1.0):
        raise ValueError("Parametr damping musi należeć do przedziału (0,1].")

    T = float(params["T"])
    N = int(params["N"])
    N_h = int(params["N_h"])
    H_max = float(params["H_max"])

    zeta = float(params["zeta"])
    gamma = float(params["gamma"])
    z_max = float(params["z_max"])
    eps_z = float(params["eps_z"])

    dt = T / N
    dH = H_max / N_h

    t_grid = np.linspace(0.0, T, N + 1)
    H_grid = np.linspace(0.0, H_max, N_h + 1)

    S = leakage_probability(H_grid, params, variant)
    premium_values = premium(H_grid, S, params)

    V_next = -utility(H_grid)

    z_policy = np.zeros((N + 1, N_h + 1), dtype=np.float32)
    V_policy = (
        np.empty((N + 1, N_h + 1), dtype=np.float32)
        if save_value_policy
        else None
    )

    if V_policy is not None:
        V_policy[-1] = V_next.astype(np.float32)

    iteration_counts = np.zeros(N, dtype=np.int32)
    policy_errors = np.zeros(N, dtype=np.float64)

    z_guess = np.zeros(N_h + 1, dtype=np.float64)

    for n in range(N - 1, -1, -1):
        if n < N - 1:
            z_guess = z_policy[n + 1].astype(np.float64, copy=True)

        converged = False
        V_current = None
        final_error = math.inf

        for policy_iter in range(1, max_policy_iter + 1):
            V_current = solve_fixed_policy_step(
                V_next=V_next,
                z=z_guess,
                H=H_grid,
                dH=dH,
                dt=dt,
                S=S,
                premium_values=premium_values,
                lambda_poisson=lambda_poisson,
                params=params,
            )

            V_H = spatial_derivative(V_current, dH)

            z_update = np.clip(
                (-V_H - zeta) / gamma,
                0.0,
                z_max,
            )

            final_error = float(np.max(np.abs(z_update - z_guess)))

            if final_error <= eps_z:
                z_guess = z_update
                converged = True
                break

            z_guess = (
                (1.0 - damping) * z_guess
                + damping * z_update
            )

        if not converged:
            raise RuntimeError(
                f"Brak zbieżności iteracji polityki dla n={n}. "
                f"Ostatni błąd={final_error:.6e}."
            )

        V_current = solve_fixed_policy_step(
            V_next=V_next,
            z=z_guess,
            H=H_grid,
            dH=dH,
            dt=dt,
            S=S,
            premium_values=premium_values,
            lambda_poisson=lambda_poisson,
            params=params,
        )

        z_policy[n] = z_guess.astype(np.float32)

        if V_policy is not None:
            V_policy[n] = V_current.astype(np.float32)

        iteration_counts[n] = policy_iter
        policy_errors[n] = final_error
        V_next = V_current

        if n % 100 == 0 or n == N - 1:
            print(
                f"  krok n={n:4d}/{N}, "
                f"iteracje={policy_iter:2d}, "
                f"błąd={final_error:.3e}"
            )

    H0 = float(params["H_0"])
    k0 = int(np.clip(round(H0 / dH), 0, N_h))

    return {
        "t_grid": t_grid,
        "H_grid": H_grid,
        "z_policy": z_policy,
        "V_policy": V_policy,
        "V_initial": float(V_next[k0]),
        "z_initial": float(z_policy[0, k0]),
        "iteration_counts": iteration_counts,
        "policy_errors": policy_errors,
        "dt": dt,
        "dH": dH,
        "H0_index": k0,
    }

def save_surface_plot(
    t_grid: np.ndarray,
    H_grid: np.ndarray,
    values: np.ndarray,
    zlabel: str,
    output_path: Path,
) -> None:
    t_idx = np.linspace(0, len(t_grid) - 1, min(201, len(t_grid))).astype(int)
    h_idx = np.linspace(0, len(H_grid) - 1, min(101, len(H_grid))).astype(int)

    T_mesh, H_mesh = np.meshgrid(
        t_grid[t_idx],
        H_grid[h_idx],
        indexing="ij",
    )
    Z = values[np.ix_(t_idx, h_idx)]

    fig = plt.figure(figsize=(9.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        T_mesh,
        H_mesh,
        Z,
        cmap="Blues",
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$H$")
    ax.set_zlabel(zlabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_results(
    params: dict,
    variant: str,
    lambda_poisson: float,
    expected_count: float,
    result: dict,
    output_root: Path,
) -> Path:
    output_dir = output_root / f"full_grid_results_Poisson_{variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    z_path = output_dir / "z_policy_full_float32.npy"
    np.save(z_path, result["z_policy"])

    V_path = None
    if result["V_policy"] is not None:
        V_path = output_dir / "V_policy_full_float32.npy"
        np.save(V_path, result["V_policy"])

    np.save(output_dir / "t_grid.npy", result["t_grid"])
    np.save(output_dir / "H_grid.npy", result["H_grid"])
    np.save(output_dir / "policy_iteration_counts.npy", result["iteration_counts"])

    params_out = dict(params)
    params_out.update({
        "model_type": "homogeneous_poisson",
        "leakage_variant": variant,
        "poisson_lambda": lambda_poisson,
        "matched_expected_count": expected_count,
        "policy_shape": list(result["z_policy"].shape),
    })

    with (output_dir / "full_solver.params.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(params_out, file, indent=2, ensure_ascii=False)

    with (output_dir / "full_solver_params.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(params_out, file, indent=2, ensure_ascii=False)

    summary = {
        "variant": variant,
        "model_type": "homogeneous_poisson",
        "stable": True,
        "T": float(params["T"]),
        "N": int(params["N"]),
        "N_h": int(params["N_h"]),
        "dt": result["dt"],
        "dH": result["dH"],
        "poisson_lambda": lambda_poisson,
        "matched_expected_count": expected_count,
        "V_initial_grid_point": result["V_initial"],
        "z_initial_grid_point": result["z_initial"],
        "min_z_0": float(np.min(result["z_policy"][0])),
        "max_z_0": float(np.max(result["z_policy"][0])),
        "mean_z_0": float(np.mean(result["z_policy"][0])),
        "share_z_max_0": float(
            np.mean(
                np.isclose(
                    result["z_policy"][0],
                    float(params["z_max"]),
                    atol=1e-6,
                )
            )
        ),
        "share_z_zero_0": float(
            np.mean(np.isclose(result["z_policy"][0], 0.0, atol=1e-8))
        ),
        "max_policy_iterations": int(np.max(result["iteration_counts"])),
        "mean_policy_iterations": float(np.mean(result["iteration_counts"])),
        "max_final_policy_error": float(np.max(result["policy_errors"])),
        "z_policy_file": str(z_path),
        "V_policy_file": str(V_path) if V_path is not None else None,
    }

    with (output_dir / "full_solver_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    save_surface_plot(
        t_grid=result["t_grid"],
        H_grid=result["H_grid"],
        values=result["z_policy"],
        zlabel=r"$z^*(t,H)$",
        output_path=output_dir / "z_policy_t_H_poisson.png",
    )

    if result["V_policy"] is not None:
        save_surface_plot(
            t_grid=result["t_grid"],
            H_grid=result["H_grid"],
            values=result["V_policy"],
            zlabel=r"$V(t,H)$",
            output_path=output_dir / "V_t_H_poisson.png",
        )

    return output_dir

def run_variant(
    params: dict,
    variant: str,
    lambda_poisson: float,
    expected_count: float,
    args: argparse.Namespace,
) -> None:
    print(f"\n=== Rozwiązywanie modelu Poissona: {variant} ===")
    print(f"lambda_P = {lambda_poisson:.9f}")
    print(f"E[N_T]   = {expected_count:.9f}")

    start = time.time()

    result = solve_poisson_policy(
        params=params,
        variant=variant,
        lambda_poisson=lambda_poisson,
        max_policy_iter=args.max_policy_iter,
        damping=args.damping,
        save_value_policy=args.save_value_policy,
    )

    output_dir = save_results(
        params=params,
        variant=variant,
        lambda_poisson=lambda_poisson,
        expected_count=expected_count,
        result=result,
        output_root=args.output_root,
    )

    elapsed = time.time() - start

    print("\nPodsumowanie:")
    print(f"V(0,H_0) = {result['V_initial']:.6f}")
    print(f"z*(0,H_0)= {result['z_initial']:.6f}")
    print(
        "iteracje polityki: "
        f"średnia={np.mean(result['iteration_counts']):.2f}, "
        f"maksimum={np.max(result['iteration_counts'])}"
    )
    print(f"Czas obliczeń: {elapsed:.2f} s")
    print(f"Wyniki: {output_dir.resolve()}")

def main() -> None:
    args = parse_args()
    params = load_params(args.params)

    expected_count = expected_hawkes_count(params)

    if args.poisson_lambda is None:
        lambda_poisson = expected_count / float(params["T"])
    else:
        lambda_poisson = float(args.poisson_lambda)

    if lambda_poisson <= 0:
        raise ValueError("Intensywność Poissona musi być dodatnia.")

    variants = ("S1", "S2") if args.variant == "both" else (args.variant,)

    print("=== Pełna polityka dla jednorodnego procesu Poissona ===")
    print(f"Parametry: {args.params.resolve()}")
    print(f"E_Hawkes[N_T] = {expected_count:.9f}")
    print(f"lambda_P      = {lambda_poisson:.9f}")

    for variant in variants:
        run_variant(
            params=params,
            variant=variant,
            lambda_poisson=lambda_poisson,
            expected_count=expected_count,
            args=args,
        )

    print("\nWszystkie obliczenia zakończono poprawnie.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nBŁĄD: {error}", file=sys.stderr)
        sys.exit(1)
