"""Solve the full-grid S1 policy for the no-insurance benchmark."""

import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.format import open_memmap

from grids import build_grids
from solver import (
    initialize_terminal_value,
    initialize_strategy,
    solve_policy_iteration_step,
)


PARAMS_PATH = "model_params.json"
OUTPUT_DIR = "insurance_impact_results_fixed_eta80_zmax100/no_insurance_eta80_zmax100"

FULL_N = 1000
FULL_N_LAMBDA = 200
FULL_N_H = 200

LEAKAGE_FUNCTION = "exp"

SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75]

PRINT_EVERY = 100

OVERWRITE = True

def load_params(path: str = PARAMS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)

    return params

def prepare_params(base_params: dict) -> dict:
    params = deepcopy(base_params)

    params["N"] = int(FULL_N)
    params["N_lambda"] = int(FULL_N_LAMBDA)
    params["N_h"] = int(FULL_N_H)

    params["r"] = 0.0
    params["pi_0"] = 0.0
    params["eta_d"] = 80.0
    params["z_max"] = 100.0

    return params

def make_output_dir(path: str = OUTPUT_DIR) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))

def snapshot_indices_from_times(params: dict, grids: dict) -> dict:
    """Map selected snapshot times to time-grid indices."""
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

def time_label(t: float) -> str:
    return f"t_{t:.3f}".replace(".", "_")

def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_surface_csv(path: Path, t_grid: np.ndarray, x_grid: np.ndarray, values: np.ndarray) -> None:
    """Save a time surface to CSV; the first column is time."""
    columns = ["t"] + [f"{x:.6g}" for x in x_grid]
    data = np.column_stack([t_grid, values])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(path, index=False)

def save_heatmap_csv(path: Path, lambda_grid: np.ndarray, H_grid: np.ndarray, values: np.ndarray) -> None:
    """Save a lambda-H matrix to CSV with H in rows and lambda in columns."""
    df = pd.DataFrame(values.T, index=H_grid, columns=lambda_grid)
    df.index.name = "H"
    df.to_csv(path)

def plot_heatmap(
    values: np.ndarray,
    grids: dict,
    title: str,
    colorbar_label: str,
    output_path_png: Path,
    output_path_pdf: Path,
) -> None:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    fig, ax = plt.subplots(figsize=(8.5, 6))

    im = ax.imshow(
        values.T,
        origin="lower",
        extent=[lambda_grid[0], lambda_grid[-1], H_grid[0], H_grid[-1]],
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Poziom zabezpieczeń $H$")

    fig.tight_layout()
    fig.savefig(output_path_png, dpi=300)
    fig.savefig(output_path_pdf)
    plt.close(fig)

def plot_snapshot_slices(
    snapshots: dict,
    grids: dict,
    params: dict,
    output_dir: Path,
    variable_name: str,
    ylabel: str,
) -> None:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for t, arr in sorted(snapshots.items()):
        ax.plot(H_grid, arr[j0, :], linewidth=2.0, label=rf"$t={t:.2f}$")
    ax.set_title(rf"Przekroje ${variable_name}(t,\lambda_0,H)$")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{variable_name}_snapshot_slices_vs_H_lambda0.png", dpi=300)
    fig.savefig(output_dir / f"{variable_name}_snapshot_slices_vs_H_lambda0.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for t, arr in sorted(snapshots.items()):
        ax.plot(lambda_grid, arr[:, k0], linewidth=2.0, label=rf"$t={t:.2f}$")
    ax.set_title(rf"Przekroje ${variable_name}(t,\lambda,H_0)$")
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{variable_name}_snapshot_slices_vs_lambda_H0.png", dpi=300)
    fig.savefig(output_dir / f"{variable_name}_snapshot_slices_vs_lambda_H0.pdf")
    plt.close(fig)

def run_full_solver() -> None:
    base_params = load_params(PARAMS_PATH)
    params = prepare_params(base_params)
    grids = build_grids(params)
    output_dir = make_output_dir(OUTPUT_DIR)

    t_grid = grids["t"]
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    N = int(params["N"])
    N_lambda = int(params["N_lambda"])
    N_h = int(params["N_h"])
    z_max = float(params["z_max"])

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])
    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    snapshot_indices = snapshot_indices_from_times(params, grids)

    print("\n" + "=" * 80)
    print("Pełny solver S1: wariant bez ubezpieczenia")
    print("=" * 80)
    print(f"Folder wyników: {output_dir.resolve()}")
    print(f"N = {N}, N_lambda = {N_lambda}, N_h = {N_h}")
    print(f"dt = {grids['dt']}, d_lambda = {grids['d_lambda']}, dH = {grids['dH']}")
    print(
        f"a_1 = {params['a_1']}, a_2 = {params['a_2']}, b = {params['b']}, "
        f"v = {params['v']}, pi_0 = {params['pi_0']}, r = {params['r']}, "
        f"eta_d = {params['eta_d']}, gamma = {params['gamma']}"
    )
    print(f"Funkcja wycieku: {LEAKAGE_FUNCTION}")
    print(f"Indeks lambda_0: j0 = {j0}, lambda[j0] = {lambda_grid[j0]}")
    print(f"Indeks H_0: k0 = {k0}, H[k0] = {H_grid[k0]}")
    print("Snapshoty:", sorted(snapshot_indices.values()))

    save_json(params, output_dir / "full_solver_params.json")
    pd.DataFrame({"t": t_grid}).to_csv(output_dir / "t_grid.csv", index=False)
    pd.DataFrame({"lambda": lambda_grid}).to_csv(output_dir / "lambda_grid.csv", index=False)
    pd.DataFrame({"H": H_grid}).to_csv(output_dir / "H_grid.csv", index=False)

    z_policy_path = output_dir / "z_policy_full_float32.npy"
    if z_policy_path.exists() and OVERWRITE:
        z_policy_path.unlink()

    z_policy = open_memmap(
        z_policy_path,
        mode="w+",
        dtype="float32",
        shape=(N, N_lambda + 1, N_h + 1),
    )

    metadata = {
        "description": "Polityka S1 dla wariantu bez ubezpieczenia: z_policy_full_float32[n, j, k] = z^*(t_n, lambda_j, H_k)",
        "shape": [N, N_lambda + 1, N_h + 1],
        "dtype": "float32",
        "time_indices": "n=0,...,N-1 correspond to t_grid[n]",
        "terminal_time": "No policy is stored for t=T; use zero control at terminal time if needed.",
    }
    save_json(metadata, output_dir / "z_policy_full_metadata.json")

    z_surface_t_H_lambda0 = np.zeros((N + 1, N_h + 1), dtype=np.float64)
    z_surface_t_lambda_H0 = np.zeros((N + 1, N_lambda + 1), dtype=np.float64)
    V_surface_t_H_lambda0 = np.zeros((N + 1, N_h + 1), dtype=np.float64)
    V_surface_t_lambda_H0 = np.zeros((N + 1, N_lambda + 1), dtype=np.float64)

    z_snapshots = {}
    V_snapshots = {}

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    V_surface_t_H_lambda0[N, :] = V_next[j0, :]
    V_surface_t_lambda_H0[N, :] = V_next[:, k0]
    z_surface_t_H_lambda0[N, :] = 0.0
    z_surface_t_lambda_H0[N, :] = 0.0

    if N in snapshot_indices:
        t_snapshot = snapshot_indices[N]
        V_snapshots[t_snapshot] = V_next.copy()
        z_snapshots[t_snapshot] = np.zeros_like(V_next)

    stable = True
    last_step = 0
    diagnostics_rows = []

    print("\nRozpoczynam iterację wsteczną...")

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

        z_policy[current_n, :, :] = z_current.astype("float32")

        V_surface_t_H_lambda0[current_n, :] = V_current[j0, :]
        V_surface_t_lambda_H0[current_n, :] = V_current[:, k0]
        z_surface_t_H_lambda0[current_n, :] = z_current[j0, :]
        z_surface_t_lambda_H0[current_n, :] = z_current[:, k0]

        if current_n in snapshot_indices:
            t_snapshot = snapshot_indices[current_n]
            V_snapshots[t_snapshot] = V_current.copy()
            z_snapshots[t_snapshot] = z_current.copy()
            print(f"Zapisano snapshot dla t = {t_snapshot:.3f}")

        if step == 0 or (step + 1) % PRINT_EVERY == 0 or step == N - 1:
            row = {
                "step": step + 1,
                "n": current_n,
                "t": current_t,
                "iterations": iterations,
                "diff_V": diff_V,
                "diff_z": diff_z,
                "min_V": float(V_current.min()),
                "max_V": float(V_current.max()),
                "mean_V": float(V_current.mean()),
                "min_z": float(z_current.min()),
                "max_z": float(z_current.max()),
                "mean_z": float(z_current.mean()),
                "share_z_max": share_z_max,
                "share_z_zero": share_z_zero,
            }
            diagnostics_rows.append(row)

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

    z_policy.flush()

    V_0 = V_next
    z_0 = z_start

    save_surface_csv(output_dir / "V_surface_t_H_lambda0.csv", t_grid, H_grid, V_surface_t_H_lambda0)
    save_surface_csv(output_dir / "V_surface_t_lambda_H0.csv", t_grid, lambda_grid, V_surface_t_lambda_H0)
    save_surface_csv(output_dir / "z_surface_t_H_lambda0.csv", t_grid, H_grid, z_surface_t_H_lambda0)
    save_surface_csv(output_dir / "z_surface_t_lambda_H0.csv", t_grid, lambda_grid, z_surface_t_lambda_H0)

    snapshot_npz_data = {
        "lambda_grid": lambda_grid,
        "H_grid": H_grid,
    }

    for t, V in sorted(V_snapshots.items()):
        label = time_label(t)
        snapshot_npz_data[f"V_{label}"] = V
        save_heatmap_csv(output_dir / f"V_heatmap_data_{label}.csv", lambda_grid, H_grid, V)
        plot_heatmap(
            V,
            grids,
            title=rf"Funkcja wartości $V(t,\lambda,H)$ dla $t={t:.2f}$",
            colorbar_label=r"Funkcja wartości $V$",
            output_path_png=output_dir / f"V_heatmap_{label}.png",
            output_path_pdf=output_dir / f"V_heatmap_{label}.pdf",
        )

    for t, z in sorted(z_snapshots.items()):
        label = time_label(t)
        snapshot_npz_data[f"z_{label}"] = z
        save_heatmap_csv(output_dir / f"z_heatmap_data_{label}.csv", lambda_grid, H_grid, z)
        plot_heatmap(
            z,
            grids,
            title=rf"Strategia optymalna $z^*(t,\lambda,H)$ dla $t={t:.2f}$",
            colorbar_label=r"Optymalna intensywność inwestycji $z^*$",
            output_path_png=output_dir / f"z_heatmap_{label}.png",
            output_path_pdf=output_dir / f"z_heatmap_{label}.pdf",
        )

    np.savez_compressed(output_dir / "V_z_snapshots.npz", **snapshot_npz_data)

    plot_snapshot_slices(V_snapshots, grids, params, output_dir, "V", r"Funkcja wartości $V$")
    plot_snapshot_slices(z_snapshots, grids, params, output_dir, "z", r"Optymalna intensywność inwestycji $z^*$")

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(output_dir / "full_solver_diagnostics.csv", index=False)

    summary = {
        "stable": bool(stable),
        "last_step": int(last_step),
        "N": int(N),
        "N_lambda": int(N_lambda),
        "N_h": int(N_h),
        "dt": float(grids["dt"]),
        "d_lambda": float(grids["d_lambda"]),
        "dH": float(grids["dH"]),
        "lambda_0": lambda_0,
        "H_0": H_0,
        "j0": int(j0),
        "k0": int(k0),
        "V_initial_grid_point": float(V_0[j0, k0]),
        "z_initial_grid_point": float(z_0[j0, k0]),
        "min_V_0": float(V_0.min()),
        "max_V_0": float(V_0.max()),
        "mean_V_0": float(V_0.mean()),
        "min_z_0": float(z_0.min()),
        "max_z_0": float(z_0.max()),
        "mean_z_0": float(z_0.mean()),
        "share_z_max_0": float(np.mean(z_0 >= z_max - 1e-8)),
        "share_z_zero_0": float(np.mean(z_0 <= 1e-8)),
        "z_policy_file": str(z_policy_path),
    }

    save_json(summary, output_dir / "full_solver_summary.json")
    pd.DataFrame([summary]).to_csv(output_dir / "full_solver_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("Podsumowanie finalnego solvera")
    print("=" * 80)
    print(pd.DataFrame([summary]).to_string(index=False))

    print("\nNajważniejsze zapisane pliki:")
    print(f"- {output_dir / 'z_policy_full_float32.npy'}")
    print(f"- {output_dir / 'V_surface_t_H_lambda0.csv'}")
    print(f"- {output_dir / 'V_surface_t_lambda_H0.csv'}")
    print(f"- {output_dir / 'z_surface_t_H_lambda0.csv'}")
    print(f"- {output_dir / 'z_surface_t_lambda_H0.csv'}")
    print(f"- {output_dir / 'V_heatmap_t_0_000.png'}")
    print(f"- {output_dir / 'z_heatmap_t_0_000.png'}")
    print(f"- {output_dir / 'full_solver_summary.csv'}")


if __name__ == "__main__":
    run_full_solver()
