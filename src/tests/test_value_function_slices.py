"""Save and plot value-function slices at selected time points."""

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


PARAMS_PATH = "model_params.json"
OUTPUT_DIR = "value_function_slices_results"

LEAKAGE_FUNCTION = "exp"


SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75]

PRINT_EVERY = 250


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


def snapshot_indices_from_times(params: dict, grids: dict) -> dict:
    """Zwraca mapę: indeks czasowy n -> żądany czas t."""
    T = float(params["T"])
    N = int(params["N"])
    dt = float(grids["dt"])

    index_to_time = {}

    for t in SNAPSHOT_TIMES:
        t_clipped = min(max(float(t), 0.0), T)
        n = int(round(t_clipped / dt))
        n = min(max(n, 0), N)
        index_to_time[n] = n * dt

    return index_to_time


def format_time_label(t: float) -> str:
    return f"t={t:.2f}".replace(".", "_")


def solve_and_store_value_snapshots(params: dict) -> dict:
    grids = build_grids(params)

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    N = int(params["N"])
    z_max = float(params["z_max"])

    snapshot_indices = snapshot_indices_from_times(params, grids)

    V_snapshots = {}
    z_snapshots = {}

    if N in snapshot_indices:
        t_terminal = snapshot_indices[N]
        V_snapshots[t_terminal] = V_next.copy()
        z_snapshots[t_terminal] = z_start.copy()

    print("\n" + "=" * 80)
    print("Test przekrojów funkcji wartości V")
    print("=" * 80)
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
    print("Zapisywane momenty czasu:", sorted(snapshot_indices.values()))

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

        if current_n in snapshot_indices:
            t_snapshot = snapshot_indices[current_n]
            V_snapshots[t_snapshot] = V_current.copy()
            z_snapshots[t_snapshot] = z_current.copy()
            print(f"Zapisano snapshot dla t = {t_snapshot:.3f}")

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

    return {
        "params": params,
        "grids": grids,
        "V_snapshots": V_snapshots,
        "z_snapshots": z_snapshots,
        "stable": stable,
        "last_step": last_step,
    }


def compute_value_diagnostics(result: dict) -> pd.DataFrame:
    grids = result["grids"]
    params = result["params"]
    V_snapshots = result["V_snapshots"]

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    rows = []

    for t, V in sorted(V_snapshots.items()):
        diff_lambda = np.diff(V, axis=0)
        diff_H = np.diff(V, axis=1)

        rows.append(
            {
                "t": float(t),
                "V_initial_grid_point": float(V[j0, k0]),
                "min_V": float(np.min(V)),
                "max_V": float(np.max(V)),
                "mean_V": float(np.mean(V)),
                "share_diff_lambda_nonnegative": float(np.mean(diff_lambda >= -1e-8)),
                "min_diff_lambda": float(np.min(diff_lambda)),
                "max_diff_lambda": float(np.max(diff_lambda)),
                "share_diff_H_nonpositive": float(np.mean(diff_H <= 1e-8)),
                "min_diff_H": float(np.min(diff_H)),
                "max_diff_H": float(np.max(diff_H)),
            }
        )

    return pd.DataFrame(rows)


def save_slice_data(result: dict, output_dir: Path) -> None:
    grids = result["grids"]
    params = result["params"]
    V_snapshots = result["V_snapshots"]

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    df_H = pd.DataFrame({"H": H_grid})
    for t, V in sorted(V_snapshots.items()):
        df_H[f"V_t_{t:.3f}"] = V[j0, :]
    df_H.to_csv(output_dir / "V_slices_vs_H_lambda0.csv", index=False)

    df_lambda = pd.DataFrame({"lambda": lambda_grid})
    for t, V in sorted(V_snapshots.items()):
        df_lambda[f"V_t_{t:.3f}"] = V[:, k0]
    df_lambda.to_csv(output_dir / "V_slices_vs_lambda_H0.csv", index=False)

    for t, V in sorted(V_snapshots.items()):
        label = format_time_label(t)
        pd.DataFrame(
            V.T,
            index=H_grid,
            columns=lambda_grid,
        ).to_csv(output_dir / f"V_heatmap_data_{label}.csv")


def save_npz(result: dict, output_dir: Path) -> None:
    arrays = {
        "lambda_grid": result["grids"]["lambda"],
        "H_grid": result["grids"]["H"],
    }

    for t, V in sorted(result["V_snapshots"].items()):
        label = format_time_label(t)
        arrays[f"V_{label}"] = V

    for t, z in sorted(result["z_snapshots"].items()):
        label = format_time_label(t)
        arrays[f"z_{label}"] = z

    np.savez_compressed(output_dir / "value_function_snapshots.npz", **arrays)


def plot_V_vs_H(result: dict, output_dir: Path) -> None:
    grids = result["grids"]
    params = result["params"]
    V_snapshots = result["V_snapshots"]

    H_grid = grids["H"]
    lambda_0 = float(params["lambda_0"])
    j0 = nearest_index(grids["lambda"], lambda_0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for t, V in sorted(V_snapshots.items()):
        ax.plot(
            H_grid,
            V[j0, :],
            linewidth=2.0,
            label=rf"$t={t:.2f}$",
        )

    ax.set_title(r"Przekroje funkcji wartości $V(t,\lambda_0,H)$")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Funkcja wartości $V$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "V_slices_vs_H_lambda0.png", dpi=300)
    fig.savefig(output_dir / "V_slices_vs_H_lambda0.pdf")
    plt.close(fig)


def plot_V_vs_lambda(result: dict, output_dir: Path) -> None:
    grids = result["grids"]
    params = result["params"]
    V_snapshots = result["V_snapshots"]

    lambda_grid = grids["lambda"]
    H_0 = float(params["H_0"])
    k0 = nearest_index(grids["H"], H_0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for t, V in sorted(V_snapshots.items()):
        ax.plot(
            lambda_grid,
            V[:, k0],
            linewidth=2.0,
            label=rf"$t={t:.2f}$",
        )

    ax.set_title(r"Przekroje funkcji wartości $V(t,\lambda,H_0)$")
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Funkcja wartości $V$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "V_slices_vs_lambda_H0.png", dpi=300)
    fig.savefig(output_dir / "V_slices_vs_lambda_H0.pdf")
    plt.close(fig)


def plot_V_heatmaps(result: dict, output_dir: Path) -> None:
    grids = result["grids"]
    V_snapshots = result["V_snapshots"]

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    all_min = min(float(np.min(V)) for V in V_snapshots.values())
    all_max = max(float(np.max(V)) for V in V_snapshots.values())

    for t, V in sorted(V_snapshots.items()):
        label = format_time_label(t)

        fig, ax = plt.subplots(figsize=(8.5, 6))

        im = ax.imshow(
            V.T,
            origin="lower",
            extent=[lambda_grid[0], lambda_grid[-1], H_grid[0], H_grid[-1]],
            aspect="auto",
            vmin=all_min,
            vmax=all_max,
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"Funkcja wartości $V$")

        ax.set_title(rf"Funkcja wartości $V(t,\lambda,H)$ dla $t={t:.2f}$")
        ax.set_xlabel(r"Intensywność ataków $\lambda$")
        ax.set_ylabel(r"Poziom zabezpieczeń $H$")

        fig.tight_layout()
        fig.savefig(output_dir / f"V_heatmap_{label}.png", dpi=300)
        fig.savefig(output_dir / f"V_heatmap_{label}.pdf")
        plt.close(fig)


def run_test() -> None:
    params = load_params(PARAMS_PATH)
    output_dir = make_output_dir(OUTPUT_DIR)

    result = solve_and_store_value_snapshots(params)

    diagnostics_df = compute_value_diagnostics(result)
    diagnostics_df.to_csv(output_dir / "value_function_diagnostics.csv", index=False)

    save_slice_data(result, output_dir)
    save_npz(result, output_dir)

    plot_V_vs_H(result, output_dir)
    plot_V_vs_lambda(result, output_dir)
    plot_V_heatmaps(result, output_dir)

    print("\n" + "=" * 80)
    print("Diagnostyka funkcji wartości V")
    print("=" * 80)
    print(diagnostics_df.to_string(index=False))

    print("\nZapisane pliki:")
    print(f"- {output_dir / 'value_function_diagnostics.csv'}")
    print(f"- {output_dir / 'V_slices_vs_H_lambda0.png'}")
    print(f"- {output_dir / 'V_slices_vs_lambda_H0.png'}")
    print(f"- {output_dir / 'value_function_snapshots.npz'}")
    print(f"- folder z osobnymi heatmapami V dla kolejnych momentów czasu")


if __name__ == "__main__":
    run_test()
