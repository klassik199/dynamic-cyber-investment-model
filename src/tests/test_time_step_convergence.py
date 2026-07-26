import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from grids import build_grids
from solver import (
    initialize_terminal_value,
    initialize_strategy,
    solve_policy_iteration_step,
)


# =====================================================================
# USTAWIENIA TESTU
# =====================================================================

PARAMS_PATH = "model_params.json"
OUTPUT_DIR = "time_step_convergence_results"

# Test zbieżności względem kroku czasowego.
# Siatka przestrzenna pozostaje stała, zmieniamy tylko N, czyli dt = T / N.
N_VALUES = [500, 1000, 1500]

# Stała siatka przestrzenna dla testu.
N_LAMBDA_TEST = 100
N_H_TEST = 100

LEAKAGE_FUNCTION = "exp"

# Wypis diagnostyki co około 1/4 przebiegu.
PRINT_EVERY_FRACTION = 4


# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================

def load_params(path: str = PARAMS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)

    return params


def make_output_dir(path: str = OUTPUT_DIR) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def bilinear_interpolate(
    array: np.ndarray,
    lambda_value: float,
    H_value: float,
    grids: dict,
) -> float:
    """Interpolacja dwuliniowa wartości tablicy na siatce (lambda, H)."""
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    d_lambda = float(grids["d_lambda"])
    dH = float(grids["dH"])

    N_lambda = len(lambda_grid) - 1
    N_h = len(H_grid) - 1

    lambda_value = min(max(lambda_value, float(lambda_grid[0])), float(lambda_grid[-1]))
    H_value = min(max(H_value, float(H_grid[0])), float(H_grid[-1]))

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


def prepare_params(base_params: dict, N_value: int) -> dict:
    params = deepcopy(base_params)

    params["N"] = int(N_value)
    params["N_lambda"] = int(N_LAMBDA_TEST)
    params["N_h"] = int(N_H_TEST)

    return params


# =====================================================================
# ROZWIĄZYWANIE DLA JEDNEGO N
# =====================================================================

def solve_for_N(base_params: dict, N_value: int) -> dict:
    params = prepare_params(base_params, N_value)
    grids = build_grids(params)

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    N = int(params["N"])
    z_max = float(params["z_max"])

    print("\n" + "=" * 80)
    print(f"Test kroku czasowego: N = {N}")
    print("=" * 80)
    print(
        f"dt = {grids['dt']}, "
        f"N_lambda = {params['N_lambda']}, N_h = {params['N_h']}, "
        f"d_lambda = {grids['d_lambda']}, dH = {grids['dH']}"
    )
    print(
        f"a_1 = {params['a_1']}, a_2 = {params['a_2']}, b = {params['b']}, "
        f"v = {params['v']}, r = {params['r']}, eta_d = {params['eta_d']}"
    )

    stable = True
    last_step = 0

    print_every = max(1, N // PRINT_EVERY_FRACTION)

    for step in range(N):
        current_n = N - step - 1
        current_t = current_n * grids["dt"]

        V_current, z_current, iterations, diff_V, diff_z = solve_policy_iteration_step(
            V_next=V_next,
            z_start=z_start,
            grids=grids,
            params=params,
            leakage_function=LEAKAGE_FUNCTION,
            max_iter=20,
            verbose=False,
        )

        last_step = step + 1

        max_abs_V = max(abs(float(V_current.min())), abs(float(V_current.max())))
        share_z_max = float(np.mean(z_current >= z_max - 1e-8))
        share_z_zero = float(np.mean(z_current <= 1e-8))

        if (
            iterations >= 20
            or not np.isfinite(V_current).all()
            or not np.isfinite(z_current).all()
            or diff_V > 1e3
            or diff_z > 1.0
            or max_abs_V > 1e8
        ):
            stable = False
            print(
                "UWAGA: możliwa niestabilność numeryczna. "
                f"krok={step + 1}, diff_V={diff_V:.2e}, "
                f"diff_z={diff_z:.2e}, max_abs_V={max_abs_V:.4e}"
            )
            break

        if step == 0 or (step + 1) % print_every == 0 or step == N - 1:
            print(
                f"krok {step + 1:4d}/{N}, "
                f"t = {current_t:.3f}, "
                f"iteracje = {iterations:2d}, "
                f"diff_V = {diff_V:.2e}, "
                f"diff_z = {diff_z:.2e}, "
                f"min V = {V_current.min():.4f}, "
                f"max V = {V_current.max():.4f}, "
                f"mean z = {z_current.mean():.4f}, "
                f"max z = {z_current.max():.4f}, "
                f"udział z=z_max = {100.0 * share_z_max:.2f}%, "
                f"udział z=0 = {100.0 * share_z_zero:.2f}%"
            )

        V_next = V_current
        z_start = z_current

    V_0 = V_next
    z_0 = z_start

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    V_initial_state = bilinear_interpolate(V_0, lambda_0, H_0, grids)
    z_initial_state = bilinear_interpolate(z_0, lambda_0, H_0, grids)

    summary = {
        "N": int(N),
        "dt": float(grids["dt"]),
        "N_lambda": int(params["N_lambda"]),
        "N_h": int(params["N_h"]),
        "d_lambda": float(grids["d_lambda"]),
        "dH": float(grids["dH"]),
        "stable": bool(stable),
        "last_step": int(last_step),
        "V_initial_state": float(V_initial_state),
        "z_initial_state": float(z_initial_state),
        "min_V_0": float(np.min(V_0)),
        "max_V_0": float(np.max(V_0)),
        "min_z_0": float(np.min(z_0)),
        "mean_z_0": float(np.mean(z_0)),
        "max_z_0": float(np.max(z_0)),
        "share_z_max_0": float(np.mean(z_0 >= z_max - 1e-8)),
        "share_z_zero_0": float(np.mean(z_0 <= 1e-8)),
    }

    return {
        "N": int(N),
        "params": params,
        "grids": grids,
        "V_0": V_0,
        "z_0": z_0,
        "summary": summary,
    }


# =====================================================================
# PORÓWNANIE WYNIKÓW DLA RÓŻNYCH N
# =====================================================================

def compute_pairwise_differences(results: dict) -> pd.DataFrame:
    rows = []

    sorted_N = sorted(results.keys())

    for N_prev, N_next in zip(sorted_N[:-1], sorted_N[1:]):
        z_prev = results[N_prev]["z_0"]
        z_next = results[N_next]["z_0"]

        V_prev = results[N_prev]["V_0"]
        V_next = results[N_next]["V_0"]

        diff_z = z_next - z_prev
        diff_V = V_next - V_prev

        rows.append(
            {
                "comparison": f"N={N_next} minus N={N_prev}",
                "N_prev": int(N_prev),
                "N_next": int(N_next),
                "max_abs_diff_z": float(np.max(np.abs(diff_z))),
                "mean_abs_diff_z": float(np.mean(np.abs(diff_z))),
                "max_abs_diff_V": float(np.max(np.abs(diff_V))),
                "mean_abs_diff_V": float(np.mean(np.abs(diff_V))),
                "diff_z_initial_state": float(
                    results[N_next]["summary"]["z_initial_state"]
                    - results[N_prev]["summary"]["z_initial_state"]
                ),
                "diff_V_initial_state": float(
                    results[N_next]["summary"]["V_initial_state"]
                    - results[N_prev]["summary"]["V_initial_state"]
                ),
                "diff_mean_z_0": float(
                    results[N_next]["summary"]["mean_z_0"]
                    - results[N_prev]["summary"]["mean_z_0"]
                ),
            }
        )

    return pd.DataFrame(rows)


def save_npz(results: dict, output_dir: Path) -> None:
    arrays = {}

    first = results[sorted(results.keys())[0]]
    arrays["lambda_grid"] = first["grids"]["lambda"]
    arrays["H_grid"] = first["grids"]["H"]

    for N_value, result in results.items():
        arrays[f"z_0_N{N_value}"] = result["z_0"]
        arrays[f"V_0_N{N_value}"] = result["V_0"]

    np.savez_compressed(output_dir / "time_step_convergence_arrays.npz", **arrays)


# =====================================================================
# WYKRESY
# =====================================================================

def plot_summary_metrics(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        summary_df["N"],
        summary_df["V_initial_state"],
        marker="o",
        linewidth=2.0,
        label=r"$V(0,\lambda_0,H_0)$",
    )

    ax.set_title(r"Zbieżność funkcji wartości względem liczby kroków czasowych $N$")
    ax.set_xlabel(r"Liczba kroków czasowych $N$")
    ax.set_ylabel(r"Wartość początkowa $V(0,\lambda_0,H_0)$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "time_convergence_V_initial.png", dpi=300)
    fig.savefig(output_dir / "time_convergence_V_initial.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        summary_df["N"],
        summary_df["z_initial_state"],
        marker="o",
        linewidth=2.0,
        label=r"$z^*(0,\lambda_0,H_0)$",
    )
    ax.plot(
        summary_df["N"],
        summary_df["mean_z_0"],
        marker="o",
        linewidth=2.0,
        label=r"średnie $z^*(0,\lambda,H)$",
    )

    ax.set_title(r"Zbieżność strategii względem liczby kroków czasowych $N$")
    ax.set_xlabel(r"Liczba kroków czasowych $N$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "time_convergence_z_metrics.png", dpi=300)
    fig.savefig(output_dir / "time_convergence_z_metrics.pdf")
    plt.close(fig)


def plot_H_slices(results: dict, output_dir: Path) -> None:
    first_key = sorted(results.keys())[0]
    grids = results[first_key]["grids"]
    params = results[first_key]["params"]

    H_grid = grids["H"]
    lambda_0 = float(params["lambda_0"])
    j0 = nearest_index(grids["lambda"], lambda_0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for N_value in sorted(results.keys()):
        ax.plot(
            H_grid,
            results[N_value]["z_0"][j0, :],
            linewidth=2.0,
            label=rf"$N={N_value}$",
        )

    ax.set_title(r"Przekrój strategii $z^*(0,\lambda_0,H)$ dla różnych $N$")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "time_convergence_z_vs_H_lambda0.png", dpi=300)
    fig.savefig(output_dir / "time_convergence_z_vs_H_lambda0.pdf")
    plt.close(fig)


def plot_lambda_slices(results: dict, output_dir: Path) -> None:
    first_key = sorted(results.keys())[0]
    grids = results[first_key]["grids"]
    params = results[first_key]["params"]

    lambda_grid = grids["lambda"]
    H_0 = float(params["H_0"])
    k0 = nearest_index(grids["H"], H_0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for N_value in sorted(results.keys()):
        ax.plot(
            lambda_grid,
            results[N_value]["z_0"][:, k0],
            linewidth=2.0,
            label=rf"$N={N_value}$",
        )

    ax.set_title(r"Przekrój strategii $z^*(0,\lambda,H_0)$ dla różnych $N$")
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "time_convergence_z_vs_lambda_H0.png", dpi=300)
    fig.savefig(output_dir / "time_convergence_z_vs_lambda_H0.pdf")
    plt.close(fig)


def plot_last_difference_heatmap(results: dict, output_dir: Path) -> None:
    sorted_N = sorted(results.keys())

    if len(sorted_N) < 2:
        return

    N_prev = sorted_N[-2]
    N_next = sorted_N[-1]

    grids = results[N_next]["grids"]
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    diff = results[N_next]["z_0"] - results[N_prev]["z_0"]

    max_abs = float(max(abs(np.min(diff)), abs(np.max(diff))))
    max_abs = max(max_abs, 1e-8)

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    im = ax.imshow(
        diff.T,
        origin="lower",
        extent=[lambda_grid[0], lambda_grid[-1], H_grid[0], H_grid[-1]],
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Różnica strategii $\Delta z^*$")

    ax.set_title(rf"Różnica strategii: $z^*_{{N={N_next}}}-z^*_{{N={N_prev}}}$")
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Poziom zabezpieczeń $H$")

    fig.tight_layout()
    fig.savefig(output_dir / f"heatmap_diff_z_N{N_next}_minus_N{N_prev}.png", dpi=300)
    fig.savefig(output_dir / f"heatmap_diff_z_N{N_next}_minus_N{N_prev}.pdf")
    plt.close(fig)

    pd.DataFrame(
        diff.T,
        index=H_grid,
        columns=lambda_grid,
    ).to_csv(output_dir / f"heatmap_diff_z_N{N_next}_minus_N{N_prev}.csv")


# =====================================================================
# GŁÓWNY TEST
# =====================================================================

def run_test() -> None:
    base_params = load_params(PARAMS_PATH)
    output_dir = make_output_dir(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("Test zbieżności względem kroku czasowego")
    print("=" * 80)
    print(f"Wyniki zostaną zapisane w folderze: {output_dir.resolve()}")
    print("Wartości N:", N_VALUES)
    print(
        f"Stała siatka przestrzenna: "
        f"N_lambda = {N_LAMBDA_TEST}, N_h = {N_H_TEST}"
    )

    results = {}

    for N_value in N_VALUES:
        result = solve_for_N(base_params, N_value)
        results[int(N_value)] = result

    summary_df = pd.DataFrame([results[N]["summary"] for N in sorted(results.keys())])
    summary_df.to_csv(output_dir / "time_step_convergence_summary.csv", index=False)

    diff_df = compute_pairwise_differences(results)
    diff_df.to_csv(output_dir / "time_step_convergence_pairwise_differences.csv", index=False)

    save_npz(results, output_dir)

    plot_summary_metrics(summary_df, output_dir)
    plot_H_slices(results, output_dir)
    plot_lambda_slices(results, output_dir)
    plot_last_difference_heatmap(results, output_dir)

    print("\n" + "=" * 80)
    print("Podsumowanie testu zbieżności względem N")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Różnice między kolejnymi wartościami N")
    print("=" * 80)
    print(diff_df.to_string(index=False))

    print("\nZapisane pliki:")
    print(f"- {output_dir / 'time_step_convergence_summary.csv'}")
    print(f"- {output_dir / 'time_step_convergence_pairwise_differences.csv'}")
    print(f"- {output_dir / 'time_convergence_V_initial.png'}")
    print(f"- {output_dir / 'time_convergence_z_metrics.png'}")
    print(f"- {output_dir / 'time_convergence_z_vs_H_lambda0.png'}")
    print(f"- {output_dir / 'time_convergence_z_vs_lambda_H0.png'}")
    print(f"- {output_dir / 'time_step_convergence_arrays.npz'}")


if __name__ == "__main__":
    run_test()
