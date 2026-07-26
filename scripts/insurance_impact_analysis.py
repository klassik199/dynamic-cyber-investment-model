import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import sparse
from numpy.lib.format import open_memmap

from grids import build_grids
from solver import (
    initialize_terminal_value,
    initialize_strategy,
    solve_policy_iteration_step,
)


# =====================================================================
# ANALIZA WPLYWU UBEZPIECZENIA -- WERSJA POPRAWIONA
# =====================================================================
# Skrypt:
# 1. NIE przelicza wariantu ubezpieczonego.
# 2. Wczytuje poprawny bazowy wariant z folderu INSURED_RESULTS_DIR.
# 3. Sprawdza, czy polityka ubezpieczona nie jest wyzerowana.
# 4. Liczy wariant bez ubezpieczenia dla eta_d=80 i z_max=100.
# 5. Tworzy porownanie:
#       Delta z^* = z^*_{bez ubezpieczenia} - z^*_{z ubezpieczeniem}
#
# Uruchom dopiero po zakonczeniu solvera S1 i sprawdzeniu, ze:
# full_grid_results_S1_new/z_policy_full_float32.npy nie jest zerowe.


# =====================================================================
# USTAWIENIA
# =====================================================================

PARAMS_PATH = "model_params.json"

# Folder z poprawnie przeliczonym wariantem bazowym S1.
# Jeśli po solverze zmienisz nazwę folderu na full_grid_results_S1,
# możesz tutaj zmienić ścieżkę.
INSURED_RESULTS_DIR = Path("full_grid_results_S1")

# Folder wyników tej analizy.
OUTPUT_ROOT = Path("insurance_impact_results_fixed_eta80_zmax100")
NO_INSURANCE_DIR = OUTPUT_ROOT / "no_insurance_eta80_zmax100"
COMPARISON_DIR = OUTPUT_ROOT / "comparison"

# Parametry wariantu bez ubezpieczenia.
NO_INSURANCE_UPDATES = {
    "pi_0": 0.0,
    "r": 0.0,
    "eta_d": 80.0,
    "z_max": 100.0,
}

# Pełna siatka, zgodna z wariantem bazowym.
FULL_N = 1000
FULL_N_LAMBDA = 200
FULL_N_H = 200

LEAKAGE_FUNCTION = "exp"

SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75]
PRINT_EVERY = 100
OVERWRITE_NO_INSURANCE = True


# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================

def load_params(path: str = PARAMS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_params(base_params: dict, updates: dict | None = None) -> dict:
    params = deepcopy(base_params)

    params["N"] = int(FULL_N)
    params["N_lambda"] = int(FULL_N_LAMBDA)
    params["N_h"] = int(FULL_N_H)

    if updates:
        for k, v in updates.items():
            params[k] = v

    return params


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def snapshot_index(t: float, dt: float, N: int) -> int:
    n = int(round(float(t) / float(dt)))
    return min(max(n, 0), N - 1)


def read_grid_csv(path: Path, column: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku siatki: {path.resolve()}")
    df = pd.read_csv(path)
    if column not in df.columns:
        # awaryjnie bierzemy pierwszą kolumnę
        return df.iloc[:, 0].to_numpy(dtype=float)
    return df[column].to_numpy(dtype=float)


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_policy_nonzero(policy_path: Path, label: str) -> None:
    if not policy_path.exists():
        raise FileNotFoundError(f"Brak pliku polityki dla wariantu {label}: {policy_path.resolve()}")

    z = np.load(policy_path, mmap_mode="r")
    z0_min = float(z[0].min())
    z0_max = float(z[0].max())
    z0_mean = float(z[0].mean())
    share_zero = float((z[0] <= 1e-12).mean())

    print(f"\nKontrola polityki: {label}")
    print(f"plik: {policy_path}")
    print(f"shape = {z.shape}")
    print(f"t=0 min/max/mean = {z0_min:.6f} / {z0_max:.6f} / {z0_mean:.6f}")
    print(f"share zeros t=0 = {share_zero:.6f}")

    if z0_max <= 1e-10 or share_zero > 0.99:
        raise RuntimeError(
            f"Polityka dla wariantu {label} wygląda na wyzerowaną. "
            "Przerwano, żeby nie tworzyć błędnego porównania."
        )


def save_surface_csv(path: Path, t_grid: np.ndarray, x_grid: np.ndarray, values: np.ndarray) -> None:
    columns = ["t"] + [f"{x:.6g}" for x in x_grid]
    data = np.column_stack([t_grid, values])
    pd.DataFrame(data, columns=columns).to_csv(path, index=False)


def save_heatmap_csv(path: Path, lambda_grid: np.ndarray, H_grid: np.ndarray, values: np.ndarray) -> None:
    df = pd.DataFrame(values.T, index=H_grid, columns=lambda_grid)
    df.index.name = "H"
    df.to_csv(path)


def save_excel_pl_copy(src: Path, dst: Path) -> None:
    """
    Kopia CSV wygodniejsza dla polskiego Excela:
    separator ; i przecinek dziesiętny.
    """
    df = pd.read_csv(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, sep=";", decimal=",")


def copy_csv_matrix_excel_pl(src: Path, dst: Path) -> None:
    df = pd.read_csv(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, sep=";", decimal=",")


# =====================================================================
# SOLVER DLA WARIANTU BEZ UBEZPIECZENIA
# =====================================================================

def run_no_insurance_solver(base_params: dict) -> dict:
    params = prepare_params(base_params, NO_INSURANCE_UPDATES)
    grids = build_grids(params)

    NO_INSURANCE_DIR.mkdir(parents=True, exist_ok=True)

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

    print("\n" + "=" * 80)
    print("Solver wariantu bez ubezpieczenia")
    print("=" * 80)
    print(f"Folder wyników: {NO_INSURANCE_DIR.resolve()}")
    print(f"N={N}, N_lambda={N_lambda}, N_h={N_h}")
    print(f"pi_0={params['pi_0']}, r={params['r']}, eta_d={params['eta_d']}, z_max={params['z_max']}")
    print(f"leakage_function={LEAKAGE_FUNCTION}")
    print(f"j0={j0}, lambda[j0]={lambda_grid[j0]}; k0={k0}, H[k0]={H_grid[k0]}")

    save_json(params, NO_INSURANCE_DIR / "full_solver_params.json")
    pd.DataFrame({"t": t_grid}).to_csv(NO_INSURANCE_DIR / "t_grid.csv", index=False)
    pd.DataFrame({"lambda": lambda_grid}).to_csv(NO_INSURANCE_DIR / "lambda_grid.csv", index=False)
    pd.DataFrame({"H": H_grid}).to_csv(NO_INSURANCE_DIR / "H_grid.csv", index=False)

    z_policy_path = NO_INSURANCE_DIR / "z_policy_full_float32.npy"
    if z_policy_path.exists() and OVERWRITE_NO_INSURANCE:
        z_policy_path.unlink()

    z_policy = open_memmap(
        z_policy_path,
        mode="w+",
        dtype="float32",
        shape=(N, N_lambda + 1, N_h + 1),
    )

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    diagnostics_rows = []
    stable = True
    last_step = 0

    print("\nRozpoczynam iterację wsteczną dla wariantu bez ubezpieczenia...")

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
                f"t={current_t:.3f}, "
                f"iter={iterations:2d}, "
                f"diff_V={diff_V:.2e}, "
                f"diff_z={diff_z:.2e}, "
                f"mean z={z_current.mean():.4f}, "
                f"max z={z_current.max():.4f}, "
                f"udział z=z_max={100.0 * share_z_max:.2f}%, "
                f"udział z=0={100.0 * share_z_zero:.2f}%"
            )

        V_next = V_current
        z_start = z_current

    z_policy.flush()

    V_0 = V_next
    z_0 = z_start

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(NO_INSURANCE_DIR / "full_solver_diagnostics.csv", index=False)

    summary = {
        "variant": "bez_ubezpieczenia",
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
        "pi_0": float(params["pi_0"]),
        "r": float(params["r"]),
        "eta_d": float(params["eta_d"]),
        "z_max": float(params["z_max"]),
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

    save_json(summary, NO_INSURANCE_DIR / "full_solver_summary.json")
    pd.DataFrame([summary]).to_csv(NO_INSURANCE_DIR / "full_solver_summary.csv", index=False)

    print("\nPodsumowanie wariantu bez ubezpieczenia:")
    print(pd.DataFrame([summary]).to_string(index=False))

    return summary


# =====================================================================
# POROWNANIE WARIANTOW
# =====================================================================

def make_comparison(base_params: dict, no_summary: dict) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    excel_dir = COMPARISON_DIR / "excel_pl"
    excel_dir.mkdir(parents=True, exist_ok=True)

    insured_policy_path = INSURED_RESULTS_DIR / "z_policy_full_float32.npy"
    no_policy_path = NO_INSURANCE_DIR / "z_policy_full_float32.npy"

    validate_policy_nonzero(insured_policy_path, "z ubezpieczeniem")
    validate_policy_nonzero(no_policy_path, "bez ubezpieczenia")

    z_ins = np.load(insured_policy_path, mmap_mode="r")
    z_no = np.load(no_policy_path, mmap_mode="r")

    if z_ins.shape != z_no.shape:
        raise ValueError(f"Niezgodne rozmiary polityk: insured={z_ins.shape}, no={z_no.shape}")

    # Siatki bierzemy z wariantu ubezpieczonego, bo to jest baseline.
    t_grid = read_grid_csv(INSURED_RESULTS_DIR / "t_grid.csv", "t")
    lambda_grid = read_grid_csv(INSURED_RESULTS_DIR / "lambda_grid.csv", "lambda")
    H_grid = read_grid_csv(INSURED_RESULTS_DIR / "H_grid.csv", "H")

    N = z_ins.shape[0]
    dt = float(t_grid[1] - t_grid[0]) if len(t_grid) > 1 else float(base_params["T"]) / N

    lambda_0 = float(base_params["lambda_0"])
    H_0 = float(base_params["H_0"])
    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    insured_summary = load_summary(INSURED_RESULTS_DIR / "full_solver_summary.json")

    # Jeżeli summary nie zawiera parametrów, uzupełniamy je z model_params.
    insured_params = load_summary(INSURED_RESULTS_DIR / "full_solver_params.json")
    if not insured_params:
        insured_params = prepare_params(base_params, None)

    insured_row = {
        "wariant": "z ubezpieczeniem",
        "pi_0": float(insured_params.get("pi_0", base_params.get("pi_0", np.nan))),
        "r": float(insured_params.get("r", base_params.get("r", np.nan))),
        "eta_d": float(insured_params.get("eta_d", base_params.get("eta_d", np.nan))),
        "z_max": float(insured_params.get("z_max", base_params.get("z_max", np.nan))),
        "V_initial_grid_point": float(insured_summary.get("V_initial_grid_point", np.nan)),
        "z_initial_grid_point": float(z_ins[0, j0, k0]),
        "mean_z_0": float(z_ins[0].mean()),
        "max_z_0": float(z_ins[0].max()),
        "share_z_max_0": float(np.mean(z_ins[0] >= float(insured_params.get("z_max", 100.0)) - 1e-8)),
        "share_z_zero_0": float(np.mean(z_ins[0] <= 1e-8)),
    }

    no_row = {
        "wariant": "bez ubezpieczenia",
        "pi_0": float(no_summary["pi_0"]),
        "r": float(no_summary["r"]),
        "eta_d": float(no_summary["eta_d"]),
        "z_max": float(no_summary["z_max"]),
        "V_initial_grid_point": float(no_summary["V_initial_grid_point"]),
        "z_initial_grid_point": float(no_summary["z_initial_grid_point"]),
        "mean_z_0": float(no_summary["mean_z_0"]),
        "max_z_0": float(no_summary["max_z_0"]),
        "share_z_max_0": float(no_summary["share_z_max_0"]),
        "share_z_zero_0": float(no_summary["share_z_zero_0"]),
    }

    summary_df = pd.DataFrame([insured_row, no_row])
    summary_path = COMPARISON_DIR / "insurance_impact_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(COMPARISON_DIR / "insurance_impact_summary_excel_pl.csv", index=False, sep=";", decimal=",")

    # Tabela LaTeX.
    tex = []
    tex.append(r"\begin{center}")
    tex.append(r"\textbf{Tabela X. Porównanie wariantu z ubezpieczeniem i bez ubezpieczenia}")
    tex.append(r"\vspace{0.5em}")
    tex.append("")
    tex.append(r"\renewcommand{\arraystretch}{1.25}")
    tex.append(r"\begin{tabular}{|c|c|c|c|c|c|}")
    tex.append(r"\hline")
    tex.append(r"\textbf{Wariant} & \textbf{\(\pi_0\)} & \textbf{\(r\)} & \textbf{\(\eta_d\)} & \textbf{\(V(0,\underline{\lambda},\underline{h})\)} & \textbf{\(z^*(0,\underline{\lambda},\underline{h})\)} \\")
    tex.append(r"\hline")
    for _, row in summary_df.iterrows():
        tex.append(
            f"{row['wariant']} & "
            f"{row['pi_0']:.0f} & "
            f"{row['r']:.2f} & "
            f"{row['eta_d']:.0f} & "
            f"{row['V_initial_grid_point']:.3f} & "
            f"{row['z_initial_grid_point']:.3f} \\\\"
        )
        tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{center}")
    (COMPARISON_DIR / "insurance_impact_table.tex").write_text("\n".join(tex), encoding="utf-8")

    # Różnice polityk.
    selected_indices = [snapshot_index(t, dt, N) for t in SNAPSHOT_TIMES]
    selected_indices = sorted(set(selected_indices))

    diff_summaries = []

    # Heatmapy 2x2 dla różnicy.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    all_diffs_for_scale = []
    for n in selected_indices:
        all_diffs_for_scale.append(np.asarray(z_no[n] - z_ins[n], dtype=float))
    global_min = min(float(d.min()) for d in all_diffs_for_scale)
    global_max = max(float(d.max()) for d in all_diffs_for_scale)
    abs_lim = max(abs(global_min), abs(global_max), 1e-8)

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red",
        ["#1c4673", "#f5f5f0", "#96301f"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-abs_lim, vcenter=0.0, vmax=abs_lim)

    last_im = None
    for ax, n in zip(axes, selected_indices):
        t = n * dt
        diff = np.asarray(z_no[n] - z_ins[n], dtype=float)

        save_heatmap_csv(
            COMPARISON_DIR / f"delta_z_heatmap_t_{t:.3f}.csv",
            lambda_grid,
            H_grid,
            diff,
        )
        copy_csv_matrix_excel_pl(
            COMPARISON_DIR / f"delta_z_heatmap_t_{t:.3f}.csv",
            excel_dir / f"delta_z_heatmap_t_{t:.3f}_excel_pl.csv",
        )

        diff_summaries.append({
            "t": float(t),
            "n": int(n),
            "min_delta_z": float(diff.min()),
            "max_delta_z": float(diff.max()),
            "mean_delta_z": float(diff.mean()),
            "mean_abs_delta_z": float(np.mean(np.abs(diff))),
            "share_positive": float(np.mean(diff > 1e-8)),
            "share_negative": float(np.mean(diff < -1e-8)),
            "share_zero": float(np.mean(np.abs(diff) <= 1e-8)),
        })

        last_im = ax.imshow(
            diff.T,
            origin="lower",
            extent=[lambda_grid[0], lambda_grid[-1], H_grid[0], H_grid[-1]],
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(rf"$t={t:.2f}$")
        ax.set_xlabel(r"Intensywność ataków $\lambda$")
        ax.set_ylabel(r"Poziom zabezpieczeń $H$")

    # Jeśli liczba osi > liczba snapshotów.
    for ax in axes[len(selected_indices):]:
        ax.axis("off")

    cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.90)
    cbar.set_label(r"$\Delta z^*$")

    fig.suptitle(
        r"Różnica strategii $\Delta z^* = z^*_{\mathrm{bez\ ub.}} - z^*_{\mathrm{ub.}}$",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.savefig(COMPARISON_DIR / "delta_z_heatmaps_4_times.png", dpi=300)
    fig.savefig(COMPARISON_DIR / "delta_z_heatmaps_4_times.pdf")
    plt.close(fig)

    diff_summary_df = pd.DataFrame(diff_summaries)
    diff_summary_path = COMPARISON_DIR / "insurance_impact_difference_summary.csv"
    diff_summary_df.to_csv(diff_summary_path, index=False)
    diff_summary_df.to_csv(COMPARISON_DIR / "insurance_impact_difference_summary_excel_pl.csv", index=False, sep=";", decimal=",")

    # Powierzchnie różnic do Excela: t-H przy lambda_0 oraz t-lambda przy H_0.
    # Polityka ma N warstw czasu, więc używamy t_grid[:N].
    t_policy_grid = t_grid[:N]

    delta_t_H_lambda0 = np.zeros((N, len(H_grid)), dtype=float)
    delta_t_lambda_H0 = np.zeros((N, len(lambda_grid)), dtype=float)

    for n in range(N):
        diff_n = z_no[n] - z_ins[n]
        delta_t_H_lambda0[n, :] = diff_n[j0, :]
        delta_t_lambda_H0[n, :] = diff_n[:, k0]

    save_surface_csv(COMPARISON_DIR / "delta_z_surface_t_H_lambda0.csv", t_policy_grid, H_grid, delta_t_H_lambda0)
    save_surface_csv(COMPARISON_DIR / "delta_z_surface_t_lambda_H0.csv", t_policy_grid, lambda_grid, delta_t_lambda_H0)

    save_excel_pl_copy(COMPARISON_DIR / "delta_z_surface_t_H_lambda0.csv", excel_dir / "delta_z_surface_t_H_lambda0_excel_pl.csv")
    save_excel_pl_copy(COMPARISON_DIR / "delta_z_surface_t_lambda_H0.csv", excel_dir / "delta_z_surface_t_lambda_H0_excel_pl.csv")

    # Downsample do Excela/wykresów.
    t_idx = np.linspace(0, N - 1, 101, dtype=int)
    H_idx = np.linspace(0, len(H_grid) - 1, 101, dtype=int)
    lam_idx = np.linspace(0, len(lambda_grid) - 1, 101, dtype=int)

    save_surface_csv(
        COMPARISON_DIR / "delta_z_surface_t_H_lambda0_downsampled.csv",
        t_policy_grid[t_idx],
        H_grid[H_idx],
        delta_t_H_lambda0[np.ix_(t_idx, H_idx)],
    )
    save_surface_csv(
        COMPARISON_DIR / "delta_z_surface_t_lambda_H0_downsampled.csv",
        t_policy_grid[t_idx],
        lambda_grid[lam_idx],
        delta_t_lambda_H0[np.ix_(t_idx, lam_idx)],
    )

    save_excel_pl_copy(
        COMPARISON_DIR / "delta_z_surface_t_H_lambda0_downsampled.csv",
        excel_dir / "delta_z_surface_t_H_lambda0_downsampled_excel_pl.csv",
    )
    save_excel_pl_copy(
        COMPARISON_DIR / "delta_z_surface_t_lambda_H0_downsampled.csv",
        excel_dir / "delta_z_surface_t_lambda_H0_downsampled_excel_pl.csv",
    )

    # Prosty wykres przekrojów po H dla lambda_0.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#1f4e79", "#4f7f73", "#b06a37", "#8f3f3f"]
    for idx, n in enumerate(selected_indices):
        t = n * dt
        diff = np.asarray(z_no[n] - z_ins[n], dtype=float)
        ax.plot(H_grid, diff[j0, :], linewidth=2.2, color=colors[idx % len(colors)], label=rf"$t={t:.2f}$")

    ax.axhline(0, linewidth=1.0, color="black", alpha=0.5)
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"$\Delta z^*$", rotation=0, labelpad=25)
    ax.set_title(rf"Przekroje $\Delta z^*(t,\underline{{\lambda}},H)$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "delta_z_slices_t_by_H_lambda0.png", dpi=300)
    fig.savefig(COMPARISON_DIR / "delta_z_slices_t_by_H_lambda0.pdf")
    plt.close(fig)

    # Metadane.
    comparison_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "definition": "Delta z = z_no_insurance - z_insured",
        "insured_results_dir": str(INSURED_RESULTS_DIR),
        "no_insurance_dir": str(NO_INSURANCE_DIR),
        "comparison_dir": str(COMPARISON_DIR),
        "no_insurance_updates": NO_INSURANCE_UPDATES,
        "leakage_function": LEAKAGE_FUNCTION,
        "snapshot_times": SNAPSHOT_TIMES,
        "j0": int(j0),
        "k0": int(k0),
        "lambda_0": float(lambda_grid[j0]),
        "H_0": float(H_grid[k0]),
        "global_color_abs_limit": float(abs_lim),
    }
    save_json(comparison_metadata, COMPARISON_DIR / "comparison_metadata.json")

    print("\n" + "=" * 80)
    print("Zakończono porównanie wpływu ubezpieczenia")
    print("=" * 80)
    print("\nTabela porównawcza:")
    print(summary_df.to_string(index=False))
    print("\nPodsumowanie różnic:")
    print(diff_summary_df.to_string(index=False))

    print("\nNajważniejsze pliki:")
    print(f"- {COMPARISON_DIR / 'insurance_impact_summary_excel_pl.csv'}")
    print(f"- {COMPARISON_DIR / 'insurance_impact_table.tex'}")
    print(f"- {COMPARISON_DIR / 'insurance_impact_difference_summary_excel_pl.csv'}")
    print(f"- {COMPARISON_DIR / 'delta_z_heatmaps_4_times.png'}")
    print(f"- {COMPARISON_DIR / 'delta_z_slices_t_by_H_lambda0.png'}")
    print(f"- {COMPARISON_DIR / 'excel_pl'}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("Analiza wpływu ubezpieczenia -- wersja poprawiona")
    print("=" * 80)

    base_params = load_params(PARAMS_PATH)

    # Kontrola folderu bazowego.
    if not INSURED_RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Nie znaleziono folderu z wariantem ubezpieczonym: {INSURED_RESULTS_DIR.resolve()}\n"
            "Uruchom najpierw solver S1 albo popraw INSURED_RESULTS_DIR w skrypcie."
        )

    validate_policy_nonzero(INSURED_RESULTS_DIR / "z_policy_full_float32.npy", "z ubezpieczeniem")

    # Kontrola zgodności podstawowych parametrów wariantu ubezpieczonego.
    insured_params = load_summary(INSURED_RESULTS_DIR / "full_solver_params.json")
    if insured_params:
        print("\nParametry wariantu ubezpieczonego wczytane z full_solver_params.json:")
        for key in ["pi_0", "r", "eta_d", "z_max", "a_1", "v", "N", "N_lambda", "N_h"]:
            if key in insured_params:
                print(f"{key}: {insured_params[key]}")

    # Liczymy wariant bez ubezpieczenia.
    no_summary = run_no_insurance_solver(base_params)

    # Porównanie.
    make_comparison(base_params, no_summary)


if __name__ == "__main__":
    main()
