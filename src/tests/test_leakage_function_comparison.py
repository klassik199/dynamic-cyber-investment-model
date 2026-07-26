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
OUTPUT_DIR = "leakage_function_comparison_results"

SCENARIOS = [
    {
        "key": "S1_exp",
        "label": r"$S_1(H,v)=v e^{-a_1H}$",
        "leakage_function": "exp",
    },
    {
        "key": "S2_rational",
        "label": r"$S_2(H,v)=\frac{v}{(1+a_2H)^b}$",
        "leakage_function": "rational",
    },
]

# Testowa siatka przestrzenna.
N_LAMBDA_TEST = 100
N_H_TEST = 100

# Liczba kroków czasowych zostaje z model_params.json.
N_TEST = None

PRINT_EVERY = 250


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


def prepare_params(base_params: dict) -> dict:
    params = deepcopy(base_params)

    params["N_lambda"] = int(N_LAMBDA_TEST)
    params["N_h"] = int(N_H_TEST)

    if N_TEST is not None:
        params["N"] = int(N_TEST)

    return params


# =====================================================================
# ROZWIĄZYWANIE MODELU DLA DANEJ FUNKCJI S
# =====================================================================

def solve_leakage_scenario(base_params: dict, scenario: dict) -> dict:
    params = prepare_params(base_params)
    grids = build_grids(params)

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    N = int(params["N"])
    z_max = float(params["z_max"])
    leakage_function = scenario["leakage_function"]

    print("\n" + "=" * 80)
    print(f"Scenariusz funkcji wycieku: {scenario['key']}")
    print("=" * 80)
    print(f"Funkcja: {scenario['label']}")
    print(
        f"N = {params['N']}, N_lambda = {params['N_lambda']}, N_h = {params['N_h']}"
    )
    print(
        f"dt = {grids['dt']}, d_lambda = {grids['d_lambda']}, dH = {grids['dH']}"
    )
    print(
        f"a_1 = {params['a_1']}, a_2 = {params['a_2']}, b = {params['b']}, "
        f"v = {params['v']}, pi_0 = {params['pi_0']}, r = {params['r']}, "
        f"eta_d = {params['eta_d']}, gamma = {params['gamma']}"
    )

    stable = True
    last_step = 0

    for step in range(N):
        current_n = N - step - 1
        current_t = current_n * grids["dt"]

        V_current, z_current, iterations, diff_V, diff_z = solve_policy_iteration_step(
            V_next=V_next,
            z_start=z_start,
            grids=grids,
            params=params,
            leakage_function=leakage_function,
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

        if step == 0 or (step + 1) % PRINT_EVERY == 0 or step == N - 1:
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
        "scenario_key": scenario["key"],
        "scenario_label": scenario["label"],
        "leakage_function": leakage_function,
        "N": int(params["N"]),
        "N_lambda": int(params["N_lambda"]),
        "N_h": int(params["N_h"]),
        "dt": float(grids["dt"]),
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
        "scenario": scenario,
        "params": params,
        "grids": grids,
        "V_0": V_0,
        "z_0": z_0,
        "summary": summary,
    }


# =====================================================================
# PORÓWNANIA I ZAPIS WYNIKÓW
# =====================================================================

def compute_difference_summary(results: dict) -> pd.DataFrame:
    z_exp = results["S1_exp"]["z_0"]
    z_rat = results["S2_rational"]["z_0"]

    V_exp = results["S1_exp"]["V_0"]
    V_rat = results["S2_rational"]["V_0"]

    diff_z = z_rat - z_exp
    diff_V = V_rat - V_exp

    row = {
        "comparison": "S2_rational minus S1_exp",
        "max_abs_diff_z": float(np.max(np.abs(diff_z))),
        "mean_abs_diff_z": float(np.mean(np.abs(diff_z))),
        "min_diff_z": float(np.min(diff_z)),
        "max_diff_z": float(np.max(diff_z)),
        "mean_diff_z": float(np.mean(diff_z)),
        "max_abs_diff_V": float(np.max(np.abs(diff_V))),
        "mean_abs_diff_V": float(np.mean(np.abs(diff_V))),
        "min_diff_V": float(np.min(diff_V)),
        "max_diff_V": float(np.max(diff_V)),
        "mean_diff_V": float(np.mean(diff_V)),
        "diff_z_initial_state": float(
            results["S2_rational"]["summary"]["z_initial_state"]
            - results["S1_exp"]["summary"]["z_initial_state"]
        ),
        "diff_V_initial_state": float(
            results["S2_rational"]["summary"]["V_initial_state"]
            - results["S1_exp"]["summary"]["V_initial_state"]
        ),
        "diff_mean_z_0": float(
            results["S2_rational"]["summary"]["mean_z_0"]
            - results["S1_exp"]["summary"]["mean_z_0"]
        ),
    }

    return pd.DataFrame([row])


def save_csv_outputs(results: dict, output_dir: Path) -> None:
    grids = results["S1_exp"]["grids"]
    params = results["S1_exp"]["params"]

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    df_H = pd.DataFrame(
        {
            "H": H_grid,
            "z_S1_exp": results["S1_exp"]["z_0"][j0, :],
            "z_S2_rational": results["S2_rational"]["z_0"][j0, :],
            "diff_z_S2_minus_S1": (
                results["S2_rational"]["z_0"][j0, :]
                - results["S1_exp"]["z_0"][j0, :]
            ),
            "V_S1_exp": results["S1_exp"]["V_0"][j0, :],
            "V_S2_rational": results["S2_rational"]["V_0"][j0, :],
            "diff_V_S2_minus_S1": (
                results["S2_rational"]["V_0"][j0, :]
                - results["S1_exp"]["V_0"][j0, :]
            ),
        }
    )
    df_H.to_csv(output_dir / "S1_vs_S2_slices_vs_H_lambda0.csv", index=False)

    df_lambda = pd.DataFrame(
        {
            "lambda": lambda_grid,
            "z_S1_exp": results["S1_exp"]["z_0"][:, k0],
            "z_S2_rational": results["S2_rational"]["z_0"][:, k0],
            "diff_z_S2_minus_S1": (
                results["S2_rational"]["z_0"][:, k0]
                - results["S1_exp"]["z_0"][:, k0]
            ),
            "V_S1_exp": results["S1_exp"]["V_0"][:, k0],
            "V_S2_rational": results["S2_rational"]["V_0"][:, k0],
            "diff_V_S2_minus_S1": (
                results["S2_rational"]["V_0"][:, k0]
                - results["S1_exp"]["V_0"][:, k0]
            ),
        }
    )
    df_lambda.to_csv(output_dir / "S1_vs_S2_slices_vs_lambda_H0.csv", index=False)

    diff_z = results["S2_rational"]["z_0"] - results["S1_exp"]["z_0"]
    diff_V = results["S2_rational"]["V_0"] - results["S1_exp"]["V_0"]

    pd.DataFrame(diff_z.T, index=H_grid, columns=lambda_grid).to_csv(
        output_dir / "S1_vs_S2_diff_z_heatmap_data.csv"
    )

    pd.DataFrame(diff_V.T, index=H_grid, columns=lambda_grid).to_csv(
        output_dir / "S1_vs_S2_diff_V_heatmap_data.csv"
    )


def save_npz(results: dict, output_dir: Path) -> None:
    arrays = {
        "lambda_grid": results["S1_exp"]["grids"]["lambda"],
        "H_grid": results["S1_exp"]["grids"]["H"],
        "z_S1_exp": results["S1_exp"]["z_0"],
        "z_S2_rational": results["S2_rational"]["z_0"],
        "V_S1_exp": results["S1_exp"]["V_0"],
        "V_S2_rational": results["S2_rational"]["V_0"],
        "diff_z_S2_minus_S1": results["S2_rational"]["z_0"] - results["S1_exp"]["z_0"],
        "diff_V_S2_minus_S1": results["S2_rational"]["V_0"] - results["S1_exp"]["V_0"],
    }

    np.savez_compressed(output_dir / "S1_vs_S2_arrays.npz", **arrays)


# =====================================================================
# WYKRESY
# =====================================================================

def plot_z_vs_H(results: dict, output_dir: Path) -> None:
    grids = results["S1_exp"]["grids"]
    params = results["S1_exp"]["params"]

    H_grid = grids["H"]
    lambda_0 = float(params["lambda_0"])
    j0 = nearest_index(grids["lambda"], lambda_0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        H_grid,
        results["S1_exp"]["z_0"][j0, :],
        linewidth=2.5,
        label=r"$S_1(H,v)=ve^{-a_1H}$",
    )
    ax.plot(
        H_grid,
        results["S2_rational"]["z_0"][j0, :],
        linewidth=2.5,
        label=r"$S_2(H,v)=\frac{v}{(1+a_2H)^b}$",
    )

    ax.set_title(r"Porównanie strategii $z^*(0,\lambda_0,H)$ dla funkcji $S_1$ i $S_2$")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "S1_vs_S2_z_vs_H_lambda0.png", dpi=300)
    fig.savefig(output_dir / "S1_vs_S2_z_vs_H_lambda0.pdf")
    plt.close(fig)


def plot_z_vs_lambda(results: dict, output_dir: Path) -> None:
    grids = results["S1_exp"]["grids"]
    params = results["S1_exp"]["params"]

    lambda_grid = grids["lambda"]
    H_0 = float(params["H_0"])
    k0 = nearest_index(grids["H"], H_0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        lambda_grid,
        results["S1_exp"]["z_0"][:, k0],
        linewidth=2.5,
        label=r"$S_1(H,v)=ve^{-a_1H}$",
    )
    ax.plot(
        lambda_grid,
        results["S2_rational"]["z_0"][:, k0],
        linewidth=2.5,
        label=r"$S_2(H,v)=\frac{v}{(1+a_2H)^b}$",
    )

    ax.set_title(r"Porównanie strategii $z^*(0,\lambda,H_0)$ dla funkcji $S_1$ i $S_2$")
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "S1_vs_S2_z_vs_lambda_H0.png", dpi=300)
    fig.savefig(output_dir / "S1_vs_S2_z_vs_lambda_H0.pdf")
    plt.close(fig)


def plot_V_vs_H(results: dict, output_dir: Path) -> None:
    grids = results["S1_exp"]["grids"]
    params = results["S1_exp"]["params"]

    H_grid = grids["H"]
    lambda_0 = float(params["lambda_0"])
    j0 = nearest_index(grids["lambda"], lambda_0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        H_grid,
        results["S1_exp"]["V_0"][j0, :],
        linewidth=2.5,
        label=r"$S_1(H,v)=ve^{-a_1H}$",
    )
    ax.plot(
        H_grid,
        results["S2_rational"]["V_0"][j0, :],
        linewidth=2.5,
        label=r"$S_2(H,v)=\frac{v}{(1+a_2H)^b}$",
    )

    ax.set_title(r"Porównanie funkcji wartości $V(0,\lambda_0,H)$ dla funkcji $S_1$ i $S_2$")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Funkcja wartości $V$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "S1_vs_S2_V_vs_H_lambda0.png", dpi=300)
    fig.savefig(output_dir / "S1_vs_S2_V_vs_H_lambda0.pdf")
    plt.close(fig)


def plot_difference_heatmap(
    diff: np.ndarray,
    grids: dict,
    title: str,
    colorbar_label: str,
    filename_stem: str,
    output_dir: Path,
) -> None:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

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
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Poziom zabezpieczeń $H$")

    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_stem}.png", dpi=300)
    fig.savefig(output_dir / f"{filename_stem}.pdf")
    plt.close(fig)


def make_all_plots(results: dict, output_dir: Path) -> None:
    grids = results["S1_exp"]["grids"]

    diff_z = results["S2_rational"]["z_0"] - results["S1_exp"]["z_0"]
    diff_V = results["S2_rational"]["V_0"] - results["S1_exp"]["V_0"]

    plot_z_vs_H(results, output_dir)
    plot_z_vs_lambda(results, output_dir)
    plot_V_vs_H(results, output_dir)

    plot_difference_heatmap(
        diff=diff_z,
        grids=grids,
        title=r"Różnica strategii: $z^*_{S_2}-z^*_{S_1}$",
        colorbar_label=r"Różnica strategii $\Delta z^*$",
        filename_stem="S1_vs_S2_diff_z_heatmap",
        output_dir=output_dir,
    )

    plot_difference_heatmap(
        diff=diff_V,
        grids=grids,
        title=r"Różnica funkcji wartości: $V_{S_2}-V_{S_1}$",
        colorbar_label=r"Różnica funkcji wartości $\Delta V$",
        filename_stem="S1_vs_S2_diff_V_heatmap",
        output_dir=output_dir,
    )


# =====================================================================
# GŁÓWNY TEST
# =====================================================================

def run_test() -> None:
    base_params = load_params(PARAMS_PATH)
    output_dir = make_output_dir(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("Test porównujący funkcje wycieku S_1 i S_2")
    print("=" * 80)
    print(f"Wyniki zostaną zapisane w folderze: {output_dir.resolve()}")
    print(
        f"Siatka testowa: N_lambda = {N_LAMBDA_TEST}, N_h = {N_H_TEST}. "
        f"N = {base_params['N'] if N_TEST is None else N_TEST}"
    )
    print(
        f"Parametry funkcji wycieku: "
        f"a_1 = {base_params['a_1']}, "
        f"a_2 = {base_params['a_2']}, "
        f"b = {base_params['b']}, "
        f"v = {base_params['v']}"
    )

    results = {}

    for scenario in SCENARIOS:
        result = solve_leakage_scenario(base_params, scenario)
        results[scenario["key"]] = result

    summary_df = pd.DataFrame(
        [results[key]["summary"] for key in ["S1_exp", "S2_rational"]]
    )
    summary_df.to_csv(output_dir / "S1_vs_S2_summary.csv", index=False)

    diff_df = compute_difference_summary(results)
    diff_df.to_csv(output_dir / "S1_vs_S2_difference_summary.csv", index=False)

    save_csv_outputs(results, output_dir)
    save_npz(results, output_dir)
    make_all_plots(results, output_dir)

    print("\n" + "=" * 80)
    print("Podsumowanie scenariuszy S_1 i S_2")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Różnice: S_2 minus S_1")
    print("=" * 80)
    print(diff_df.to_string(index=False))

    print("\nZapisane pliki:")
    print(f"- {output_dir / 'S1_vs_S2_summary.csv'}")
    print(f"- {output_dir / 'S1_vs_S2_difference_summary.csv'}")
    print(f"- {output_dir / 'S1_vs_S2_z_vs_H_lambda0.png'}")
    print(f"- {output_dir / 'S1_vs_S2_z_vs_lambda_H0.png'}")
    print(f"- {output_dir / 'S1_vs_S2_V_vs_H_lambda0.png'}")
    print(f"- {output_dir / 'S1_vs_S2_diff_z_heatmap.png'}")
    print(f"- {output_dir / 'S1_vs_S2_diff_V_heatmap.png'}")
    print(f"- {output_dir / 'S1_vs_S2_arrays.npz'}")


if __name__ == "__main__":
    run_test()
