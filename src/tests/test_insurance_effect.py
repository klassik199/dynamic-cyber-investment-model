"""Compare selected insurance scenarios using the initial optimal policy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def make_output_dir(path: str = "insurance_effect_results") -> Path:
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


def solve_scenario(
    params_base: dict,
    scenario_key: str,
    scenario_label: str,
    pi_0: float,
    r: float,
    eta_d: float,
) -> tuple[dict, dict, np.ndarray, dict]:
    params = params_base.copy()


    params["N"] = 1000
    params["N_lambda"] = 100
    params["N_h"] = 100

    params["z_max"] = 100
    params["H_max"] = 100
    params["gamma"] = 0.3


    params["pi_0"] = pi_0
    params["r"] = r
    params["eta_d"] = eta_d

    grids = build_grids(params)

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    print("\n" + "=" * 80)
    print(f"Scenariusz: {scenario_label}")
    print("=" * 80)
    print(f"pi_0 = {pi_0}, r = {r}, eta_d = {eta_d}")
    print(f"N = {params['N']}, N_lambda = {params['N_lambda']}, N_h = {params['N_h']}")
    print(f"dt = {grids['dt']}, d_lambda = {grids['d_lambda']}, dH = {grids['dH']}")

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
            current_t = params["T"] - (step + 1) * grids["dt"]
            share_z_max = (z_current >= params["z_max"] - 1e-8).mean()
            share_z_zero = (z_current <= 1e-8).mean()

            print(
                f"krok {step + 1:4d}/{params['N']}, "
                f"t = {current_t:.3f}, "
                f"iteracje = {iterations:2d}, "
                f"diff_V = {diff_V:.2e}, "
                f"diff_z = {diff_z:.2e}, "
                f"min V = {V_current.min():.4f}, "
                f"max V = {V_current.max():.4f}, "
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

    result = {
        "scenario_key": scenario_key,
        "scenario_label": scenario_label,
        "pi_0": float(pi_0),
        "r": float(r),
        "eta_d": float(eta_d),
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

    return params, grids, z_start.copy(), result


def plot_z_vs_H(
    grids: dict,
    scenario_z: dict,
    scenario_labels: dict,
    params: dict,
    output_dir: Path,
) -> None:
    H_grid = grids["H"]
    lambda_grid = grids["lambda"]

    lambda_0 = float(params["lambda_0"])
    j0 = nearest_index(lambda_grid, lambda_0)

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, z_policy in scenario_z.items():
        ax.plot(
            H_grid,
            z_policy[j0, :],
            linewidth=2,
            label=scenario_labels[key],
        )

    ax.set_title(r"Strategia $z^*(0,\lambda_0,H)$ w scenariuszach ubezpieczenia")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
    ax.set_ylim(0, params["z_max"] * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "insurance_z_vs_H_lambda0.png", dpi=300)
    fig.savefig(output_dir / "insurance_z_vs_H_lambda0.pdf")
    plt.close(fig)


def plot_z_vs_lambda(
    grids: dict,
    scenario_z: dict,
    scenario_labels: dict,
    params: dict,
    output_dir: Path,
) -> None:
    H_grid = grids["H"]
    lambda_grid = grids["lambda"]

    H_0 = float(params["H_0"])
    k0 = nearest_index(H_grid, H_0)

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, z_policy in scenario_z.items():
        ax.plot(
            lambda_grid,
            z_policy[:, k0],
            linewidth=2,
            label=scenario_labels[key],
        )

    ax.set_title(r"Strategia $z^*(0,\lambda,H_0)$ w scenariuszach ubezpieczenia")
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
    ax.set_ylim(0, params["z_max"] * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "insurance_z_vs_lambda_H0.png", dpi=300)
    fig.savefig(output_dir / "insurance_z_vs_lambda_H0.pdf")
    plt.close(fig)


def plot_difference_heatmap(
    grids: dict,
    difference: np.ndarray,
    title: str,
    output_name: str,
    output_dir: Path,
) -> None:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    max_abs = float(np.max(np.abs(difference)))
    if max_abs == 0:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(
        difference.T,
        origin="lower",
        aspect="auto",
        extent=[
            lambda_grid[0],
            lambda_grid[-1],
            H_grid[0],
            H_grid[-1],
        ],
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_title(title)
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Poziom zabezpieczeń $H$")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"Różnica strategii $\Delta z^*$")

    fig.tight_layout()
    fig.savefig(output_dir / f"{output_name}.png", dpi=300)
    fig.savefig(output_dir / f"{output_name}.pdf")
    plt.close(fig)


def save_slices_to_csv(
    grids: dict,
    scenario_z: dict,
    scenario_labels: dict,
    params: dict,
    output_dir: Path,
) -> None:
    H_grid = grids["H"]
    lambda_grid = grids["lambda"]

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    df_H = pd.DataFrame({"H": H_grid})
    df_lambda = pd.DataFrame({"lambda": lambda_grid})

    for key, z_policy in scenario_z.items():
        df_H[f"z_{key}"] = z_policy[j0, :]
        df_lambda[f"z_{key}"] = z_policy[:, k0]

    df_H.to_csv(output_dir / "insurance_slice_H_lambda0.csv", index=False)
    df_lambda.to_csv(output_dir / "insurance_slice_lambda_H0.csv", index=False)


def main() -> None:
    params_base = load_params()
    output_dir = make_output_dir()

    scenarios = [
        {
            "key": "base",
            "label": "Ubezpieczenie bazowe",
            "pi_0": 300,
            "r": 0.6,
            "eta_d": 40,
        },
        {
            "key": "no_insurance",
            "label": "Brak ubezpieczenia",
            "pi_0": 0,
            "r": 0.0,
            "eta_d": 100,
        },
    ]

    scenario_z = {}
    scenario_labels = {}
    results = []

    common_params = None
    common_grids = None

    for scenario in scenarios:
        params, grids, z_policy, result = solve_scenario(
            params_base=params_base,
            scenario_key=scenario["key"],
            scenario_label=scenario["label"],
            pi_0=scenario["pi_0"],
            r=scenario["r"],
            eta_d=scenario["eta_d"],
        )

        scenario_z[scenario["key"]] = z_policy
        scenario_labels[scenario["key"]] = scenario["label"]
        results.append(result)

        common_params = params
        common_grids = grids

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "insurance_effect_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("Podsumowanie scenariuszy ubezpieczenia")
    print("=" * 80)
    print(df_results.to_string(index=False))

    np.savez(
        output_dir / "insurance_policy_z0.npz",
        lambda_grid=common_grids["lambda"],
        H_grid=common_grids["H"],
        **{f"z_{key}": value for key, value in scenario_z.items()},
    )

    save_slices_to_csv(
        grids=common_grids,
        scenario_z=scenario_z,
        scenario_labels=scenario_labels,
        params=common_params,
        output_dir=output_dir,
    )

    plot_z_vs_H(
        grids=common_grids,
        scenario_z=scenario_z,
        scenario_labels=scenario_labels,
        params=common_params,
        output_dir=output_dir,
    )

    plot_z_vs_lambda(
        grids=common_grids,
        scenario_z=scenario_z,
        scenario_labels=scenario_labels,
        params=common_params,
        output_dir=output_dir,
    )


    diff_no_insurance = scenario_z["no_insurance"] - scenario_z["base"]

    plot_difference_heatmap(
        grids=common_grids,
        difference=diff_no_insurance,
        title=r"Efekt braku ubezpieczenia: $z^*_{\mathrm{bez\ ub.}}-z^*_{\mathrm{bazowy}}$",
        output_name="heatmap_effect_no_insurance",
        output_dir=output_dir,
    )

    print("\nWygenerowano pliki w folderze:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
