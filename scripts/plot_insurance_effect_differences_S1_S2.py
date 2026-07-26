# plot_insurance_effect_differences_S1_S2.py
#
# Tworzy po dwa wykresy różnic strategii optymalnych dla trajektorii
# reprezentatywnej i stresowej:
#
# 1) z_t^{*,0} - z_t^{*,I,r=0.6}
# 2) z_t^{*,I,r=0.6} - z_t^{*,I,r=0}
#
# Na każdym wykresie są dwie linie: S1 oraz S2.
# Kolory: S1 bordowy, S2 ciemnoróżowy.
#
# Skrypt zapisuje wykresy do:
#   - insurance_effect_differences_S1_S2/plots
#   - wykresy
# oraz dane pomocnicze do CSV.

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BASELINE_ID = 6221
DEFAULT_STRESS_ID = 57457
DEFAULT_SELECTION_DIR = Path("trajectory_selection_S1")

DEFAULT_S1_R06_DIR = Path("full_grid_results_S1")
DEFAULT_S1_R0_DIR = Path("insurance_impact_results_fixed_eta80_zmax100") / "insured_r0_pi300_eta40_zmax100"
DEFAULT_S1_NO_DIR = Path("insurance_impact_results_fixed_eta80_zmax100") / "no_insurance_eta80_zmax100"

DEFAULT_S2_R06_DIR = Path("full_grid_results_S2")
DEFAULT_S2_R0_DIR = Path("insurance_impact_results_S2_fixed_eta80_zmax100") / "insured_r0_pi300_eta40_zmax100"
DEFAULT_S2_NO_DIR = Path("insurance_impact_results_S2_fixed_eta80_zmax100") / "no_insurance_eta80_zmax100"

DEFAULT_OUT_DIR = Path("insurance_effect_differences_S1_S2")

FIGSIZE = (8.8, 5.4)
DPI = 600

COLOR_S1 = "#7A1F2B"   # bordowy
COLOR_S2 = "#D14A7A"   # ciemnoróżowy
LINEWIDTH = 2.6

COLOR_GROWTH_SHADE = "#DFC0C5"
GROWTH_SHADE_ALPHA = 0.34
GROWTH_SMOOTH_WINDOW = 21
GROWTH_MAX_GAP = 0.012
GROWTH_MIN_DURATION = 0.008
GROWTH_MIN_NET_RISE = 2.0


def load_params(policy_dir: Path) -> dict:
    candidates = [
        policy_dir / "full_solver_params.json",
        policy_dir / "full_solver.params.json",
        policy_dir / "full_solver_summary.json",
        Path("model_params.json"),
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                print(f"Wczytano parametry z: {path}")
                return json.load(f)

    raise FileNotFoundError(
        "Nie znaleziono pliku parametrów w:\n"
        + "\n".join(str(path) for path in candidates)
    )


def load_policy(policy_dir: Path) -> np.ndarray:
    path = policy_dir / "z_policy_full_float32.npy"
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono polityki: {path}")

    z_policy = np.load(path, mmap_mode="r")
    if z_policy.ndim != 3:
        raise ValueError(f"Niepoprawny wymiar polityki {path}: {z_policy.shape}")

    print(f"Wczytano politykę: {path}")
    print(f"Kształt polityki: {z_policy.shape}")
    return z_policy


def load_path_from_all_grids(
    selection_dir: Path,
    sim_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    lambda_path = selection_dir / "all_lambda_grid_float32.npy"
    N_path = selection_dir / "all_N_grid_uint16.npy"

    if not lambda_path.exists() or not N_path.exists():
        return None

    lambda_all = np.load(lambda_path, mmap_mode="r")
    N_all = np.load(N_path, mmap_mode="r")

    if sim_id < 0 or sim_id >= lambda_all.shape[0]:
        raise IndexError(
            f"simulation_id={sim_id} poza zakresem 0,...,{lambda_all.shape[0] - 1}."
        )

    lambda_grid = np.asarray(lambda_all[sim_id], dtype=float)
    N_grid = np.asarray(N_all[sim_id], dtype=int)
    dN = np.diff(N_grid).astype(int)
    return lambda_grid, N_grid, dN


def load_path_from_candidate_csv(
    selection_dir: Path,
    sim_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    candidate_dir = selection_dir / "candidate_trajectories"
    if not candidate_dir.exists():
        return None

    matches = sorted(candidate_dir.glob(f"*_sim_{sim_id:05d}.csv"))
    if not matches:
        return None

    df = pd.read_csv(matches[0])
    lambda_grid = df["lambda_t"].to_numpy(dtype=float)
    N_grid = df["N_t"].to_numpy(dtype=int)

    if "dN_interval_to_next" in df.columns:
        dN = df["dN_interval_to_next"].to_numpy(dtype=int)[:-1]
    else:
        dN = np.diff(N_grid).astype(int)

    return lambda_grid, N_grid, dN


def load_selected_path(
    selection_dir: Path,
    sim_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    loaded = load_path_from_all_grids(selection_dir, sim_id)
    source = "all_grids"

    if loaded is None:
        loaded = load_path_from_candidate_csv(selection_dir, sim_id)
        source = "candidate_csv"

    if loaded is None:
        raise FileNotFoundError(
            f"Nie znaleziono trajektorii dla simulation_id={sim_id} w {selection_dir}."
        )

    return (*loaded, source)


def bilinear_policy_value(
    z_policy: np.ndarray,
    n: int,
    lambda_value: float,
    H_value: float,
    params: dict,
) -> tuple[float, bool, bool]:
    n_t, n_lambda, n_H = z_policy.shape
    n = int(np.clip(n, 0, n_t - 1))

    lambda_max = float(params["lambda_max"])
    H_max = float(params["H_max"])

    lambda_clipped = lambda_value < 0.0 or lambda_value > lambda_max
    H_clipped = H_value < 0.0 or H_value > H_max

    lam = float(np.clip(lambda_value, 0.0, lambda_max))
    H = float(np.clip(H_value, 0.0, H_max))

    d_lambda = lambda_max / (n_lambda - 1)
    dH = H_max / (n_H - 1)

    x = lam / d_lambda
    y = H / dH

    j0 = int(np.floor(x))
    k0 = int(np.floor(y))

    if j0 >= n_lambda - 1:
        j0 = n_lambda - 2
        wx = 1.0
    else:
        wx = x - j0

    if k0 >= n_H - 1:
        k0 = n_H - 2
        wy = 1.0
    else:
        wy = y - k0

    j1 = j0 + 1
    k1 = k0 + 1

    z00 = float(z_policy[n, j0, k0])
    z10 = float(z_policy[n, j1, k0])
    z01 = float(z_policy[n, j0, k1])
    z11 = float(z_policy[n, j1, k1])

    z_low = (1.0 - wx) * z00 + wx * z10
    z_high = (1.0 - wx) * z01 + wx * z11
    z_value = (1.0 - wy) * z_low + wy * z_high

    return z_value, lambda_clipped, H_clipped


def compute_z_H_on_path(
    lambda_grid: np.ndarray,
    z_policy: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    T = float(params["T"])
    N = len(lambda_grid) - 1
    dt = T / N

    H0 = float(params["H_0"])
    rho = float(params["rho"])

    H_grid = np.empty(N + 1, dtype=float)
    z_grid = np.empty(N + 1, dtype=float)
    H_grid[0] = H0

    lambda_clip_count = 0
    H_clip_count = 0

    exp_rho = np.exp(-rho * dt) if rho != 0.0 else None

    for n in range(N):
        z_n, lambda_clipped, H_clipped = bilinear_policy_value(
            z_policy=z_policy,
            n=n,
            lambda_value=float(lambda_grid[n]),
            H_value=float(H_grid[n]),
            params=params,
        )

        z_grid[n] = z_n
        lambda_clip_count += int(lambda_clipped)
        H_clip_count += int(H_clipped)

        if rho == 0.0:
            H_grid[n + 1] = H_grid[n] + dt * z_n
        else:
            H_grid[n + 1] = exp_rho * H_grid[n] + ((1.0 - exp_rho) / rho) * z_n

    z_T, lambda_clipped, H_clipped = bilinear_policy_value(
        z_policy=z_policy,
        n=N,
        lambda_value=float(lambda_grid[-1]),
        H_value=float(H_grid[-1]),
        params=params,
    )

    z_grid[-1] = z_T
    lambda_clip_count += int(lambda_clipped)
    H_clip_count += int(H_clipped)

    diagnostics = {
        "z_0": float(z_grid[0]),
        "max_z": float(np.max(z_grid)),
        "integrated_z": float(np.sum(z_grid[:-1]) * dt),
        "H_T": float(H_grid[-1]),
        "lambda_clip_count": int(lambda_clip_count),
        "H_clip_count": int(H_clip_count),
    }

    return z_grid, H_grid, diagnostics


def find_lambda_growth_intervals(
    t: np.ndarray,
    lam: np.ndarray,
) -> list[tuple[float, float]]:
    if len(t) < 3:
        return []

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return []

    window = max(3, int(GROWTH_SMOOTH_WINDOW))
    if window % 2 == 0:
        window += 1

    smooth = (
        pd.Series(lam)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    rising = np.zeros(len(t), dtype=bool)
    rising[1:] = np.diff(smooth) > 0.0

    max_gap_steps = max(1, int(round(GROWTH_MAX_GAP / dt)))

    i = 0
    while i < len(rising):
        if rising[i]:
            i += 1
            continue

        gap_start = i
        while i < len(rising) and not rising[i]:
            i += 1
        gap_end = i

        bounded_left = gap_start > 0 and rising[gap_start - 1]
        bounded_right = gap_end < len(rising) and rising[gap_end]

        if bounded_left and bounded_right and (gap_end - gap_start) <= max_gap_steps:
            rising[gap_start:gap_end] = True

    intervals = []
    i = 0

    while i < len(rising):
        if not rising[i]:
            i += 1
            continue

        start = i
        while i + 1 < len(rising) and rising[i + 1]:
            i += 1
        end = i

        left = max(0, start - 1)
        right = min(len(t) - 1, end + 1)

        duration = t[right] - t[left]
        net_rise = smooth[right] - smooth[left]

        if duration >= GROWTH_MIN_DURATION and net_rise >= GROWTH_MIN_NET_RISE:
            intervals.append((float(t[left]), float(t[right])))

        i += 1

    return intervals


def add_growth_shading(
    ax: plt.Axes,
    intervals: list[tuple[float, float]],
) -> None:
    for left, right in intervals:
        ax.axvspan(
            left,
            right,
            facecolor=COLOR_GROWTH_SHADE,
            alpha=GROWTH_SHADE_ALPHA,
            edgecolor="none",
            zorder=0,
        )


def format_axes(ax: plt.Axes) -> None:
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color="#E8E8E8", alpha=1.0, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlim(0.0, 1.0)


def padded_ylim(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    values = values[np.isfinite(values)]

    lower = float(np.min(values))
    upper = float(np.max(values))

    if np.isclose(lower, upper):
        pad = max(abs(lower), 1.0) * 0.08
    else:
        pad = (upper - lower) * 0.08

    return lower - pad, upper + pad


def choose_legend_loc(
    y1: np.ndarray,
    y2: np.ndarray,
) -> str:
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    ymin = float(np.nanmin([np.nanmin(y1), np.nanmin(y2)]))
    ymax = float(np.nanmax([np.nanmax(y1), np.nanmax(y2)]))

    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        return "upper right"

    y1n = (y1 - ymin) / (ymax - ymin)
    y2n = (y2 - ymin) / (ymax - ymin)

    n = len(y1n)
    quarter = max(1, n // 4)

    left_occ = float(np.nanmean(np.maximum(y1n[:quarter], y2n[:quarter])))
    right_occ = float(np.nanmean(np.maximum(y1n[-quarter:], y2n[-quarter:])))

    return "upper right" if right_occ <= left_occ else "upper left"


def plot_difference(
    t_grid: np.ndarray,
    diff_S1: np.ndarray,
    diff_S2: np.ndarray,
    ylabel: str,
    out_path: Path,
    intervals: list[tuple[float, float]],
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    add_growth_shading(ax, intervals)

    line_s1, = ax.plot(
        t_grid,
        diff_S1,
        color=COLOR_S1,
        linewidth=LINEWIDTH,
        label=r"$S_1$",
        zorder=3,
    )

    line_s2, = ax.plot(
        t_grid,
        diff_S2,
        color=COLOR_S2,
        linewidth=LINEWIDTH,
        label=r"$S_2$",
        zorder=3,
    )

    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", zorder=2)

    ax.set_xlabel(r"$t$", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_ylim(*ylim)
    format_axes(ax)

    ax.legend(
        handles=[line_s1, line_s2],
        labels=[r"$S_1$", r"$S_2$"],
        loc=choose_legend_loc(diff_S1, diff_S2),
        fontsize=17,
        frameon=True,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tworzy wykresy różnic strategii ubezpieczeniowych dla S1 i S2."
    )

    parser.add_argument("--baseline-id", type=int, default=DEFAULT_BASELINE_ID)
    parser.add_argument("--stress-id", type=int, default=DEFAULT_STRESS_ID)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)

    parser.add_argument("--s1-r06-dir", type=Path, default=DEFAULT_S1_R06_DIR)
    parser.add_argument("--s1-r0-dir", type=Path, default=DEFAULT_S1_R0_DIR)
    parser.add_argument("--s1-no-dir", type=Path, default=DEFAULT_S1_NO_DIR)

    parser.add_argument("--s2-r06-dir", type=Path, default=DEFAULT_S2_R06_DIR)
    parser.add_argument("--s2-r0-dir", type=Path, default=DEFAULT_S2_R0_DIR)
    parser.add_argument("--s2-no-dir", type=Path, default=DEFAULT_S2_NO_DIR)

    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()

    print("=== Różnice strategii: efekty ubezpieczenia dla S1 i S2 ===")

    variant_dirs = {
        "S1_no": args.s1_no_dir,
        "S1_r0": args.s1_r0_dir,
        "S1_r06": args.s1_r06_dir,
        "S2_no": args.s2_no_dir,
        "S2_r0": args.s2_r0_dir,
        "S2_r06": args.s2_r06_dir,
    }

    params = {}
    policies = {}

    for key, path in variant_dirs.items():
        print(f"\nWariant {key}: {path}")
        params[key] = load_params(path)
        policies[key] = load_policy(path)

    plots_dir = args.out_dir / "plots"
    data_dir = args.out_dir / "data"
    wykresy_dir = Path("wykresy")

    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    wykresy_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        ("baseline", args.baseline_id),
        ("stress", args.stress_id),
    ]

    all_diffs = []

    for group, sim_id in selected:
        print(f"\nPrzetwarzanie trajektorii: {group}, simulation_id={sim_id}")

        lambda_grid, N_grid, dN, source = load_selected_path(args.selection_dir, sim_id)
        print(f"Źródło trajektorii: {source}")

        T = float(params["S1_r06"]["T"])
        t_grid = np.linspace(0.0, T, len(lambda_grid))

        intervals = find_lambda_growth_intervals(t_grid, lambda_grid)

        dynamics = {}
        diagnostics = []

        for key in variant_dirs:
            z_grid, H_grid, diag = compute_z_H_on_path(
                lambda_grid=lambda_grid,
                z_policy=policies[key],
                params=params[key],
            )
            dynamics[key] = {"z": z_grid, "H": H_grid}

            diagnostics.append(
                {
                    "group": group,
                    "simulation_id": sim_id,
                    "variant": key,
                    **diag,
                }
            )

        diff_transfer_S1 = dynamics["S1_no"]["z"] - dynamics["S1_r06"]["z"]
        diff_transfer_S2 = dynamics["S2_no"]["z"] - dynamics["S2_r06"]["z"]

        diff_rebate_S1 = dynamics["S1_r06"]["z"] - dynamics["S1_r0"]["z"]
        diff_rebate_S2 = dynamics["S2_r06"]["z"] - dynamics["S2_r0"]["z"]

        diff_df = pd.DataFrame(
            {
                "group": group,
                "simulation_id": sim_id,
                "t": t_grid,
                "lambda_t": lambda_grid,
                "N_t": N_grid,
                "diff_transfer_S1_no_minus_r06": diff_transfer_S1,
                "diff_transfer_S2_no_minus_r06": diff_transfer_S2,
                "diff_rebate_S1_r06_minus_r0": diff_rebate_S1,
                "diff_rebate_S2_r06_minus_r0": diff_rebate_S2,
            }
        )

        diff_csv = data_dir / f"{group}_sim_{sim_id:05d}_insurance_effect_differences.csv"
        diff_df.to_csv(diff_csv, index=False)
        all_diffs.append(diff_df)

        diag_csv = data_dir / f"{group}_sim_{sim_id:05d}_diagnostics.csv"
        pd.DataFrame(diagnostics).to_csv(diag_csv, index=False)

        transfer_ylim = padded_ylim([diff_transfer_S1, diff_transfer_S2])
        rebate_ylim = padded_ylim([diff_rebate_S1, diff_rebate_S2])

        transfer_name = f"{group}_sim_{sim_id:05d}_diff_no_insurance_minus_r06_S1_S2.png"
        rebate_name = f"{group}_sim_{sim_id:05d}_diff_r06_minus_r0_S1_S2.png"

        for target_dir in [plots_dir, wykresy_dir]:
            plot_difference(
                t_grid=t_grid,
                diff_S1=diff_transfer_S1,
                diff_S2=diff_transfer_S2,
                ylabel=r"$z_t^{*,0}-z_t^{*,I,r=0{,}6}$",
                out_path=target_dir / transfer_name,
                intervals=intervals,
                ylim=transfer_ylim,
            )

            plot_difference(
                t_grid=t_grid,
                diff_S1=diff_rebate_S1,
                diff_S2=diff_rebate_S2,
                ylabel=r"$z_t^{*,r=0{,}6}-z_t^{*,r=0}$",
                out_path=target_dir / rebate_name,
                intervals=intervals,
                ylim=rebate_ylim,
            )

        print(f"Zapisano wykresy dla: {group}")

    combined_path = data_dir / "all_insurance_effect_differences.csv"
    pd.concat(all_diffs, ignore_index=True).to_csv(combined_path, index=False)

    print("\nZapisano wykresy w:")
    print(f"  {plots_dir.resolve()}")
    print(f"  {wykresy_dir.resolve()}")
    print(f"Zapisano dane zbiorcze: {combined_path.resolve()}")
    print(f"Czas wykonania: {time.time() - start:.2f} s")
    print("\nGotowe.")


if __name__ == "__main__":
    main()
