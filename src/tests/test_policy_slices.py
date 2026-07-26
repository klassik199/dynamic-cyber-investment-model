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


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def make_output_dir(path: str = "policy_slices_results") -> Path:
    output_dir = Path(path)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def solve_and_save_snapshots(params_base: dict) -> tuple[dict, dict, dict, Path]:
    params = params_base.copy()

    # Parametry testowe do ładnych przekrojów.
    params["N"] = 1000
    params["N_lambda"] = 100
    params["N_h"] = 100

    params["z_max"] = 100
    params["H_max"] = 100
    params["gamma"] = 0.3

    grids = build_grids(params)

    output_dir = make_output_dir()

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    target_times = [0.0, 0.25, 0.5, 0.75]
    target_indices = {
        int(round(t / grids["dt"])): t
        for t in target_times
    }

    snapshots = {}
    diagnostics = []

    print("Test przekrojów strategii")
    print(f"N          = {params['N']}")
    print(f"N_lambda   = {params['N_lambda']}")
    print(f"N_h        = {params['N_h']}")
    print(f"dt         = {grids['dt']}")
    print(f"d_lambda   = {grids['d_lambda']}")
    print(f"dH         = {grids['dH']}")
    print(f"Wymiar V   = {V_next.shape}")
    print(f"Zapisywane czasy: {target_times}")

    for step in range(params["N"]):
        current_n = params["N"] - step - 1
        current_t = current_n * grids["dt"]

        V_current, z_current, iterations, diff_V, diff_z = solve_policy_iteration_step(
            V_next=V_next,
            z_start=z_start,
            grids=grids,
            params=params,
            leakage_function="exp",
            max_iter=20,
            verbose=False,
        )

        if current_n in target_indices:
            t_snapshot = target_indices[current_n]
            snapshots[t_snapshot] = z_current.copy()

            share_z_max = (z_current >= params["z_max"] - 1e-8).mean()
            share_z_zero = (z_current <= 1e-8).mean()

            diagnostics.append(
                {
                    "t": t_snapshot,
                    "n": current_n,
                    "min_z": float(z_current.min()),
                    "mean_z": float(z_current.mean()),
                    "max_z": float(z_current.max()),
                    "share_z_max": float(share_z_max),
                    "share_z_zero": float(share_z_zero),
                    "iterations": iterations,
                    "diff_V": diff_V,
                    "diff_z": diff_z,
                }
            )

            print(
                f"Zapisano snapshot dla t = {t_snapshot:.2f}: "
                f"min z = {z_current.min():.4f}, "
                f"mean z = {z_current.mean():.4f}, "
                f"max z = {z_current.max():.4f}, "
                f"udział z=z_max = {share_z_max:.2%}"
            )

        if step == 0 or (step + 1) % 250 == 0:
            print(
                f"krok {step + 1:4d}/{params['N']}, "
                f"t = {current_t:.3f}, "
                f"iteracje = {iterations:2d}, "
                f"diff_V = {diff_V:.2e}, "
                f"diff_z = {diff_z:.2e}, "
                f"min V = {V_current.min():.4f}, "
                f"max V = {V_current.max():.4f}, "
                f"mean z = {z_current.mean():.4f}, "
                f"max z = {z_current.max():.4f}"
            )

        max_abs_V = max(abs(V_current.min()), abs(V_current.max()))

        if iterations == 20 or diff_V > 1e3 or diff_z > 1.0 or max_abs_V > 1e8:
            raise RuntimeError(
                "Przerwano: brak zbieżności albo niestabilność numeryczna. "
                f"Krok: {step + 1}, diff_V={diff_V}, diff_z={diff_z}, max_abs_V={max_abs_V}"
            )

        V_next = V_current
        z_start = z_current

    diagnostics_df = pd.DataFrame(diagnostics).sort_values("t")
    diagnostics_path = output_dir / "policy_slices_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False)

    np.savez(
        output_dir / "policy_snapshots.npz",
        lambda_grid=grids["lambda"],
        H_grid=grids["H"],
        **{f"z_t_{str(t).replace('.', '_')}": snapshots[t] for t in snapshots},
    )

    print(f"\nZapisano diagnostykę do: {diagnostics_path.resolve()}")
    print(f"Zapisano snapshoty do: {(output_dir / 'policy_snapshots.npz').resolve()}")

    return params, grids, snapshots, output_dir


def plot_z_vs_lambda(params: dict, grids: dict, snapshots: dict, output_dir: Path) -> None:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    H_0 = float(params["H_0"])
    k0 = nearest_index(H_grid, H_0)

    fig, ax = plt.subplots(figsize=(8, 5))

    for t in sorted(snapshots.keys()):
        z_snapshot = snapshots[t]
        ax.plot(lambda_grid, z_snapshot[:, k0], linewidth=2, label=fr"$t={t:.2f}$")

    ax.set_title(r"Przekrój strategii $z^*(t,\lambda,H_0)$")
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
    ax.set_ylim(0, params["z_max"] * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Czas")

    fig.tight_layout()

    fig.savefig(output_dir / "z_vs_lambda_H0.png", dpi=300)
    fig.savefig(output_dir / "z_vs_lambda_H0.pdf")

    plt.close(fig)


def plot_z_vs_H(params: dict, grids: dict, snapshots: dict, output_dir: Path) -> None:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    lambda_0 = float(params["lambda_0"])
    j0 = nearest_index(lambda_grid, lambda_0)

    fig, ax = plt.subplots(figsize=(8, 5))

    for t in sorted(snapshots.keys()):
        z_snapshot = snapshots[t]
        ax.plot(H_grid, z_snapshot[j0, :], linewidth=2, label=fr"$t={t:.2f}$")

    ax.set_title(r"Przekrój strategii $z^*(t,\lambda_0,H)$")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
    ax.set_ylim(0, params["z_max"] * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Czas")

    fig.tight_layout()

    fig.savefig(output_dir / "z_vs_H_lambda0.png", dpi=300)
    fig.savefig(output_dir / "z_vs_H_lambda0.pdf")

    plt.close(fig)


def plot_heatmaps(params: dict, grids: dict, snapshots: dict, output_dir: Path) -> None:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    times = sorted(snapshots.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()

    vmin = 0.0
    vmax = float(params["z_max"])

    image = None

    for ax, t in zip(axes, times):
        z_snapshot = snapshots[t]

        image = ax.imshow(
            z_snapshot.T,
            origin="lower",
            aspect="auto",
            extent=[
                lambda_grid[0],
                lambda_grid[-1],
                H_grid[0],
                H_grid[-1],
            ],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

        ax.set_title(fr"$t={t:.2f}$")
        ax.set_xlabel(r"Intensywność ataków $\lambda$")
        ax.set_ylabel(r"Poziom zabezpieczeń $H$")

    cbar = fig.colorbar(image, ax=axes, shrink=0.92)
    cbar.set_label(r"Optymalna intensywność inwestycji $z^*$")

    fig.suptitle(r"Mapy strategii $z^*(t,\lambda,H)$", fontsize=14)

    fig.savefig(output_dir / "heatmaps_z_policy.png", dpi=300)
    fig.savefig(output_dir / "heatmaps_z_policy.pdf")

    plt.close(fig)


def plot_heatmaps_separate(params: dict, grids: dict, snapshots: dict, output_dir: Path) -> None:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    for t in sorted(snapshots.keys()):
        z_snapshot = snapshots[t]

        fig, ax = plt.subplots(figsize=(7, 5))

        image = ax.imshow(
            z_snapshot.T,
            origin="lower",
            aspect="auto",
            extent=[
                lambda_grid[0],
                lambda_grid[-1],
                H_grid[0],
                H_grid[-1],
            ],
            vmin=0.0,
            vmax=float(params["z_max"]),
            cmap="viridis",
        )

        ax.set_title(fr"Strategia optymalna $z^*(t,\lambda,H)$ dla $t={t:.2f}$")
        ax.set_xlabel(r"Intensywność ataków $\lambda$")
        ax.set_ylabel(r"Poziom zabezpieczeń $H$")

        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(r"Optymalna intensywność inwestycji $z^*$")

        fig.tight_layout()

        t_label = str(t).replace(".", "_")
        fig.savefig(output_dir / f"heatmap_z_policy_t_{t_label}.png", dpi=300)
        fig.savefig(output_dir / f"heatmap_z_policy_t_{t_label}.pdf")

        plt.close(fig)


def main() -> None:
    params_base = load_params()

    params, grids, snapshots, output_dir = solve_and_save_snapshots(params_base)

    plot_z_vs_lambda(params, grids, snapshots, output_dir)
    plot_z_vs_H(params, grids, snapshots, output_dir)
    plot_heatmaps(params, grids, snapshots, output_dir)
    plot_heatmaps_separate(params, grids, snapshots, output_dir)

    print("\nWygenerowano wykresy:")
    print(output_dir / "z_vs_lambda_H0.png")
    print(output_dir / "z_vs_H_lambda0.png")
    print(output_dir / "heatmaps_z_policy.png")
    print(output_dir / "heatmap_z_policy_t_0_0.png")
    print(output_dir / "heatmap_z_policy_t_0_25.png")
    print(output_dir / "heatmap_z_policy_t_0_5.png")
    print(output_dir / "heatmap_z_policy_t_0_75.png")


if __name__ == "__main__":
    main()