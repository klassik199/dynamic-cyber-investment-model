"""Compute the full-grid HJB-PIDE policy for the rational leakage function S2."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


from grids import build_grids
from solver import (
    initialize_terminal_value,
    initialize_strategy,
    solve_policy_iteration_step,
)


PARAMS_PATH = "model_params.json"
OUTPUT_DIR = Path("full_grid_results_S2")

LEAKAGE_FUNCTION = "rational"

FORCE_N = 1000
FORCE_N_LAMBDA = 200
FORCE_N_H = 200

SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75]

PRINT_EVERY = 250

DOWNSAMPLE_ROWS = 101
DOWNSAMPLE_COLS = 101

MAKE_SURFACE_PLOTS = True

SURFACE_ELEV = 25
SURFACE_AZIM = -135

def load_params(path: str = PARAMS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)

    params["N"] = int(FORCE_N)
    params["N_lambda"] = int(FORCE_N_LAMBDA)
    params["N_h"] = int(FORCE_N_H)

    return params

def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))

def snapshot_indices_from_times(params: dict, grids: dict) -> dict:
    """Map selected snapshot times to time-grid indices."""
    dt = float(grids["dt"])
    N = int(params["N"])
    T = float(params["T"])

    index_to_time = {}
    for t in SNAPSHOT_TIMES:
        t_clipped = min(max(float(t), 0.0), T)
        n = int(round(t_clipped / dt))
        n = min(max(n, 0), N)
        index_to_time[n] = n * dt

    return index_to_time

def format_time_label(t: float) -> str:
    return f"t_{t:.3f}".replace(".", "_")

def evenly_spaced_indices(n: int, target: int) -> np.ndarray:
    if n <= target:
        return np.arange(n, dtype=int)

    idx = np.linspace(0, n - 1, target)
    idx = np.round(idx).astype(int)
    idx = np.unique(idx)

    if len(idx) < target:
        missing = target - len(idx)
        candidates = np.setdiff1d(np.arange(n), idx)
        extra = candidates[np.linspace(0, len(candidates) - 1, missing).round().astype(int)]
        idx = np.sort(np.concatenate([idx, extra]))

    return idx[:target]

def save_dataframe_both_formats(df: pd.DataFrame, path_standard: Path, path_excel: Path) -> None:
    """
    Zapisuje plik w dwóch wariantach:
    1. standardowy CSV z kropką dziesiętną,
    2. CSV dla polskiego Excela ze średnikiem i przecinkiem dziesiętnym.
    """
    path_standard.parent.mkdir(parents=True, exist_ok=True)
    path_excel.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path_standard, index=False)
    df.to_csv(path_excel, index=False, sep=";", decimal=",")

def plot_heatmap(
    Z: np.ndarray,
    lambda_grid: np.ndarray,
    H_grid: np.ndarray,
    title: str,
    colorbar_label: str,
    output_stem: Path,
    cmap: str,
    vmin=None,
    vmax=None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    im = ax.imshow(
        Z.T,
        origin="lower",
        extent=[lambda_grid[0], lambda_grid[-1], H_grid[0], H_grid[-1]],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(r"Poziom zabezpieczeń $H$")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)

def plot_slices_vs_H(
    snapshots: dict,
    H_grid: np.ndarray,
    title: str,
    ylabel: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for t, values in sorted(snapshots.items()):
        ax.plot(H_grid, values, linewidth=2.0, label=rf"$t={t:.2f}$")

    ax.set_title(title)
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)

def plot_slices_vs_lambda(
    snapshots: dict,
    lambda_grid: np.ndarray,
    title: str,
    ylabel: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for t, values in sorted(snapshots.items()):
        ax.plot(lambda_grid, values, linewidth=2.0, label=rf"$t={t:.2f}$")

    ax.set_title(title)
    ax.set_xlabel(r"Intensywność ataków $\lambda$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)

def read_surface_csv(path: Path):
    df = pd.read_csv(path)

    y = df.iloc[:, 0].astype(float).to_numpy()
    x = np.array([float(c) for c in df.columns[1:]], dtype=float)
    z = df.iloc[:, 1:].astype(float).to_numpy()

    X, Y = np.meshgrid(x, y)
    return X, Y, z

def style_3d_axis(ax):
    ax.grid(True)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))

    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.zaxis.set_tick_params(labelsize=9)

def make_surface_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    cmap: str,
    output_stem: Path,
) -> None:
    fig = plt.figure(figsize=(9.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
        rcount=101,
        ccount=101,
        shade=True,
    )

    ax.view_init(elev=SURFACE_ELEV, azim=SURFACE_AZIM)

    ax.set_title(title, pad=18, fontsize=14)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_zlabel(zlabel, labelpad=10)

    ax.set_ylim(float(np.max(Y)), float(np.min(Y)))

    style_3d_axis(ax)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.10)
    cbar.set_label(zlabel)

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)

def make_contour_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    cmap: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    contour = ax.contourf(X, Y, Z, levels=30, cmap=cmap)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.20)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(zlabel)

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)

def create_surface_plots_from_downsampled(output_dir: Path) -> None:
    input_dir = output_dir / "surfaces_downsampled" / "standard"
    plot_dir = output_dir / "surface_plots_python"
    plot_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "filename": "V_surface_t_H_lambda0.csv",
            "title": r"Funkcja wartości $V(t,\lambda_0,H)$ dla $S_2$",
            "xlabel": r"Poziom zabezpieczeń $H$",
            "ylabel": r"Czas $t$",
            "zlabel": r"Funkcja wartości $V$",
            "stem": "surface_V_t_H_lambda0",
            "cmap": "cividis",
        },
        {
            "filename": "V_surface_t_lambda_H0.csv",
            "title": r"Funkcja wartości $V(t,\lambda,H_0)$ dla $S_2$",
            "xlabel": r"Intensywność ataków $\lambda$",
            "ylabel": r"Czas $t$",
            "zlabel": r"Funkcja wartości $V$",
            "stem": "surface_V_t_lambda_H0",
            "cmap": "cividis",
        },
        {
            "filename": "z_surface_t_H_lambda0.csv",
            "title": r"Strategia optymalna $z^*(t,\lambda_0,H)$ dla $S_2$",
            "xlabel": r"Poziom zabezpieczeń $H$",
            "ylabel": r"Czas $t$",
            "zlabel": r"Optymalna intensywność inwestycji $z^*$",
            "stem": "surface_z_t_H_lambda0",
            "cmap": "viridis",
        },
        {
            "filename": "z_surface_t_lambda_H0.csv",
            "title": r"Strategia optymalna $z^*(t,\lambda,H_0)$ dla $S_2$",
            "xlabel": r"Intensywność ataków $\lambda$",
            "ylabel": r"Czas $t$",
            "zlabel": r"Optymalna intensywność inwestycji $z^*$",
            "stem": "surface_z_t_lambda_H0",
            "cmap": "viridis",
        },
    ]

    for spec in specs:
        path = input_dir / spec["filename"]
        if not path.exists():
            print(f"UWAGA: brak pliku do powierzchni: {path}")
            continue

        X, Y, Z = read_surface_csv(path)

        make_surface_plot(
            X,
            Y,
            Z,
            title=spec["title"],
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
            zlabel=spec["zlabel"],
            cmap=spec["cmap"],
            output_stem=plot_dir / spec["stem"],
        )

        make_contour_plot(
            X,
            Y,
            Z,
            title=spec["title"],
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
            zlabel=spec["zlabel"],
            cmap=spec["cmap"],
            output_stem=plot_dir / f"{spec['stem']}_contour",
        )

def downsample_surface_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 2:
        raise ValueError("Plik powierzchni musi mieć pierwszą kolumnę osi i co najmniej jedną kolumnę wartości.")

    row_idx = evenly_spaced_indices(len(df), DOWNSAMPLE_ROWS)

    first_col = df.columns[0]
    value_cols = list(df.columns[1:])
    col_idx = evenly_spaced_indices(len(value_cols), DOWNSAMPLE_COLS)
    selected_value_cols = [value_cols[i] for i in col_idx]

    out = df.loc[df.index[row_idx], [first_col] + selected_value_cols].copy()
    out.reset_index(drop=True, inplace=True)
    return out

def downsample_surface_csvs(output_dir: Path) -> None:
    surface_files = [
        "V_surface_t_H_lambda0.csv",
        "V_surface_t_lambda_H0.csv",
        "z_surface_t_H_lambda0.csv",
        "z_surface_t_lambda_H0.csv",
    ]

    standard_dir = output_dir / "surfaces_downsampled" / "standard"
    excel_dir = output_dir / "surfaces_downsampled" / "excel_pl"
    standard_dir.mkdir(parents=True, exist_ok=True)
    excel_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for filename in surface_files:
        path = output_dir / filename
        if not path.exists():
            print(f"UWAGA: pominięto przerzedzanie, brak pliku: {path}")
            continue

        df = pd.read_csv(path)
        reduced = downsample_surface_df(df)

        standard_path = standard_dir / filename
        excel_path = excel_dir / filename.replace(".csv", "_excel_pl.csv")

        reduced.to_csv(standard_path, index=False)
        reduced.to_csv(excel_path, index=False, sep=";", decimal=",")

        summary_rows.append(
            {
                "file": filename,
                "input_rows": df.shape[0],
                "input_cols": df.shape[1],
                "output_rows": reduced.shape[0],
                "output_cols": reduced.shape[1],
                "standard_csv": str(standard_path),
                "excel_pl_csv": str(excel_path),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "downsample_summary.csv", index=False)

def run_solver() -> None:
    params = load_params(PARAMS_PATH)
    grids = build_grids(params)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lambda_grid = grids["lambda"]
    H_grid = grids["H"]
    t_grid = grids["t"]

    N = int(params["N"])
    N_lambda = int(params["N_lambda"])
    N_h = int(params["N_h"])
    z_max = float(params["z_max"])

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    snapshot_indices = snapshot_indices_from_times(params, grids)

    print("=" * 80)
    print("Finalny solver na pełnej siatce — wariant S_2")
    print("=" * 80)
    print(f"Folder wynikowy: {OUTPUT_DIR.resolve()}")
    print(
        f"N = {N}, N_lambda = {N_lambda}, N_h = {N_h}, "
        f"dt = {grids['dt']}, d_lambda = {grids['d_lambda']}, dH = {grids['dH']}"
    )
    print(
        f"Funkcja wycieku: S_2(H,v)=v/(1+a_2 H)^b, "
        f"a_2 = {params['a_2']}, b = {params['b']}, v = {params['v']}"
    )
    print(
        f"pi_0 = {params['pi_0']}, r = {params['r']}, eta_d = {params['eta_d']}, "
        f"gamma = {params['gamma']}"
    )

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    z_policy_path = OUTPUT_DIR / "z_policy_full_float32.npy"
    z_policy = np.lib.format.open_memmap(
        z_policy_path,
        mode="w+",
        dtype=np.float32,
        shape=(N, N_lambda + 1, N_h + 1),
    )

    V_surface_t_H = np.zeros((N + 1, N_h + 1), dtype=np.float64)
    V_surface_t_lambda = np.zeros((N + 1, N_lambda + 1), dtype=np.float64)

    z_surface_t_H = np.zeros((N + 1, N_h + 1), dtype=np.float64)
    z_surface_t_lambda = np.zeros((N + 1, N_lambda + 1), dtype=np.float64)

    V_surface_t_H[N, :] = V_next[j0, :]
    V_surface_t_lambda[N, :] = V_next[:, k0]

    z_surface_t_H[N, :] = 0.0
    z_surface_t_lambda[N, :] = 0.0

    V_snapshots = {}
    z_snapshots = {}

    if N in snapshot_indices:
        t_snapshot = snapshot_indices[N]
        V_snapshots[t_snapshot] = V_next.copy()
        z_snapshots[t_snapshot] = z_start.copy()

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

        z_policy[current_n, :, :] = z_current.astype(np.float32)

        V_surface_t_H[current_n, :] = V_current[j0, :]
        V_surface_t_lambda[current_n, :] = V_current[:, k0]

        z_surface_t_H[current_n, :] = z_current[j0, :]
        z_surface_t_lambda[current_n, :] = z_current[:, k0]

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

    z_policy.flush()

    V_0 = V_next
    z_0 = z_start


    df_V_t_H = pd.DataFrame(V_surface_t_H, columns=[f"{x:.10g}" for x in H_grid])
    df_V_t_H.insert(0, "t", t_grid)

    df_V_t_lambda = pd.DataFrame(V_surface_t_lambda, columns=[f"{x:.10g}" for x in lambda_grid])
    df_V_t_lambda.insert(0, "t", t_grid)

    df_z_t_H = pd.DataFrame(z_surface_t_H, columns=[f"{x:.10g}" for x in H_grid])
    df_z_t_H.insert(0, "t", t_grid)

    df_z_t_lambda = pd.DataFrame(z_surface_t_lambda, columns=[f"{x:.10g}" for x in lambda_grid])
    df_z_t_lambda.insert(0, "t", t_grid)

    df_V_t_H.to_csv(OUTPUT_DIR / "V_surface_t_H_lambda0.csv", index=False)
    df_V_t_lambda.to_csv(OUTPUT_DIR / "V_surface_t_lambda_H0.csv", index=False)
    df_z_t_H.to_csv(OUTPUT_DIR / "z_surface_t_H_lambda0.csv", index=False)
    df_z_t_lambda.to_csv(OUTPUT_DIR / "z_surface_t_lambda_H0.csv", index=False)


    heatmap_dir = OUTPUT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    slice_dir = OUTPUT_DIR / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    V_min_all = min(float(np.min(V)) for V in V_snapshots.values())
    V_max_all = max(float(np.max(V)) for V in V_snapshots.values())
    z_min_all = min(float(np.min(z)) for z in z_snapshots.values())
    z_max_all = max(float(np.max(z)) for z in z_snapshots.values())

    V_slice_H = pd.DataFrame({"H": H_grid})
    V_slice_lambda = pd.DataFrame({"lambda": lambda_grid})
    z_slice_H = pd.DataFrame({"H": H_grid})
    z_slice_lambda = pd.DataFrame({"lambda": lambda_grid})

    for t, V in sorted(V_snapshots.items()):
        label = format_time_label(t)

        pd.DataFrame(V.T, index=H_grid, columns=lambda_grid).to_csv(
            heatmap_dir / f"V_heatmap_data_{label}.csv"
        )

        plot_heatmap(
            Z=V,
            lambda_grid=lambda_grid,
            H_grid=H_grid,
            title=rf"Funkcja wartości $V(t,\lambda,H)$ dla $S_2$, $t={t:.2f}$",
            colorbar_label=r"Funkcja wartości $V$",
            output_stem=heatmap_dir / f"V_heatmap_{label}",
            cmap="cividis",
            vmin=V_min_all,
            vmax=V_max_all,
        )

        V_slice_H[f"V_t_{t:.3f}"] = V[j0, :]
        V_slice_lambda[f"V_t_{t:.3f}"] = V[:, k0]

    for t, z in sorted(z_snapshots.items()):
        label = format_time_label(t)

        pd.DataFrame(z.T, index=H_grid, columns=lambda_grid).to_csv(
            heatmap_dir / f"z_heatmap_data_{label}.csv"
        )

        plot_heatmap(
            Z=z,
            lambda_grid=lambda_grid,
            H_grid=H_grid,
            title=rf"Strategia optymalna $z^*(t,\lambda,H)$ dla $S_2$, $t={t:.2f}$",
            colorbar_label=r"Optymalna intensywność inwestycji $z^*$",
            output_stem=heatmap_dir / f"z_heatmap_{label}",
            cmap="viridis",
            vmin=z_min_all,
            vmax=z_max_all,
        )

        z_slice_H[f"z_t_{t:.3f}"] = z[j0, :]
        z_slice_lambda[f"z_t_{t:.3f}"] = z[:, k0]

    V_slice_H.to_csv(slice_dir / "V_slices_vs_H_lambda0.csv", index=False)
    V_slice_lambda.to_csv(slice_dir / "V_slices_vs_lambda_H0.csv", index=False)
    z_slice_H.to_csv(slice_dir / "z_slices_vs_H_lambda0.csv", index=False)
    z_slice_lambda.to_csv(slice_dir / "z_slices_vs_lambda_H0.csv", index=False)

    plot_slices_vs_H(
        snapshots={t: V[j0, :] for t, V in V_snapshots.items()},
        H_grid=H_grid,
        title=r"Przekroje $V(t,\lambda_0,H)$ dla $S_2$",
        ylabel=r"Funkcja wartości $V$",
        output_stem=slice_dir / "V_slices_vs_H_lambda0",
    )

    plot_slices_vs_lambda(
        snapshots={t: V[:, k0] for t, V in V_snapshots.items()},
        lambda_grid=lambda_grid,
        title=r"Przekroje $V(t,\lambda,H_0)$ dla $S_2$",
        ylabel=r"Funkcja wartości $V$",
        output_stem=slice_dir / "V_slices_vs_lambda_H0",
    )

    plot_slices_vs_H(
        snapshots={t: z[j0, :] for t, z in z_snapshots.items()},
        H_grid=H_grid,
        title=r"Przekroje $z^*(t,\lambda_0,H)$ dla $S_2$",
        ylabel=r"Optymalna intensywność inwestycji $z^*$",
        output_stem=slice_dir / "z_slices_vs_H_lambda0",
    )

    plot_slices_vs_lambda(
        snapshots={t: z[:, k0] for t, z in z_snapshots.items()},
        lambda_grid=lambda_grid,
        title=r"Przekroje $z^*(t,\lambda,H_0)$ dla $S_2$",
        ylabel=r"Optymalna intensywność inwestycji $z^*$",
        output_stem=slice_dir / "z_slices_vs_lambda_H0",
    )

    arrays = {
        "lambda_grid": lambda_grid,
        "H_grid": H_grid,
        "t_grid": t_grid,
    }

    for t, V in sorted(V_snapshots.items()):
        arrays[f"V_{format_time_label(t)}"] = V

    for t, z in sorted(z_snapshots.items()):
        arrays[f"z_{format_time_label(t)}"] = z

    np.savez_compressed(OUTPUT_DIR / "snapshots_S2.npz", **arrays)

    downsample_surface_csvs(OUTPUT_DIR)

    if MAKE_SURFACE_PLOTS:
        create_surface_plots_from_downsampled(OUTPUT_DIR)

    summary = pd.DataFrame(
        [
            {
                "variant": "S2_rational",
                "leakage_function": LEAKAGE_FUNCTION,
                "stable": stable,
                "last_step": last_step,
                "N": N,
                "N_lambda": N_lambda,
                "N_h": N_h,
                "dt": float(grids["dt"]),
                "d_lambda": float(grids["d_lambda"]),
                "dH": float(grids["dH"]),
                "lambda_0": lambda_0,
                "H_0": H_0,
                "j0": j0,
                "k0": k0,
                "V_initial_grid_point": float(V_0[j0, k0]),
                "z_initial_grid_point": float(z_0[j0, k0]),
                "min_V_0": float(np.min(V_0)),
                "max_V_0": float(np.max(V_0)),
                "mean_V_0": float(np.mean(V_0)),
                "min_z_0": float(np.min(z_0)),
                "max_z_0": float(np.max(z_0)),
                "mean_z_0": float(np.mean(z_0)),
                "share_z_max_0": float(np.mean(z_0 >= z_max - 1e-8)),
                "share_z_zero_0": float(np.mean(z_0 <= 1e-8)),
                "z_policy_file": str(z_policy_path),
            }
        ]
    )

    summary.to_csv(OUTPUT_DIR / "full_grid_summary_S2.csv", index=False)

    metadata = {
        "variant": "S2_rational",
        "leakage_function": LEAKAGE_FUNCTION,
        "output_dir": str(OUTPUT_DIR),
        "params": params,
        "snapshot_times": SNAPSHOT_TIMES,
        "z_policy_file": str(z_policy_path),
    }

    with open(OUTPUT_DIR / "metadata_S2.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Podsumowanie finalnego solvera S_2")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nNajważniejsze pliki:")
    print(f"- {OUTPUT_DIR / 'z_policy_full_float32.npy'}")
    print(f"- {OUTPUT_DIR / 'V_surface_t_H_lambda0.csv'}")
    print(f"- {OUTPUT_DIR / 'V_surface_t_lambda_H0.csv'}")
    print(f"- {OUTPUT_DIR / 'z_surface_t_H_lambda0.csv'}")
    print(f"- {OUTPUT_DIR / 'z_surface_t_lambda_H0.csv'}")
    print(f"- {OUTPUT_DIR / 'surfaces_downsampled'}")
    print(f"- {OUTPUT_DIR / 'surface_plots_python'}")
    print(f"- {OUTPUT_DIR / 'heatmaps'}")
    print(f"- {OUTPUT_DIR / 'slices'}")


if __name__ == "__main__":
    run_solver()
