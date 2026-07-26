"""Compare optimal S1 trajectories with and without insurance on the same Hawkes paths."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BASELINE_ID = 6221
DEFAULT_STRESS_ID = 57457

DEFAULT_SELECTION_DIR = Path("trajectory_selection_S1")
DEFAULT_INSURED_POLICY_DIR = Path("full_grid_results_S1")
DEFAULT_UNINSURED_POLICY_DIR = Path(
    "insurance_impact_results_fixed_eta80_zmax100"
) / "no_insurance_eta80_zmax100"
DEFAULT_OUT_DIR = Path("insurance_trajectory_comparison_S1")

FIGSIZE = (8.3, 5.25)
DPI = 300

COLOR_INSURED = "#0B2F5B"
COLOR_UNINSURED = "#7A1F2B"
COLOR_SHADE = "#DFA7AE"
GRID_COLOR = "#D9D9D9"


def load_json_first_existing(paths: list[Path]) -> dict:
    for path in paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                print(f"Wczytano parametry z: {path}")
                return json.load(f)

    raise FileNotFoundError(
        "Nie znaleziono żadnego pliku parametrów:\n"
        + "\n".join(str(path) for path in paths)
    )


def load_params(policy_dir: Path) -> dict:
    return load_json_first_existing(
        [
            policy_dir / "full_solver.params.json",
            policy_dir / "full_solver_params.json",
            policy_dir / "full_solver_summary.json",
            Path("model_params.json"),
        ]
    )


def load_policy(policy_dir: Path) -> np.ndarray:
    path = policy_dir / "z_policy_full_float32.npy"
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono polityki: {path}")

    policy = np.load(path, mmap_mode="r")
    if policy.ndim != 3:
        raise ValueError(
            f"Polityka powinna mieć wymiar (t, lambda, H), "
            f"a ma kształt {policy.shape}."
        )

    print(f"Wczytano politykę: {path}")
    print(f"Kształt polityki: {policy.shape}")
    return policy


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
            f"simulation_id={sim_id} poza zakresem "
            f"0,...,{lambda_all.shape[0] - 1}."
        )

    lambda_grid = np.asarray(lambda_all[sim_id], dtype=float)
    N_grid = np.asarray(N_all[sim_id], dtype=int)
    dN = np.diff(N_grid).astype(int)

    return lambda_grid, N_grid, dN


def find_candidate_csv(selection_dir: Path, sim_id: int) -> Path | None:
    candidate_dir = selection_dir / "candidate_trajectories"
    if not candidate_dir.exists():
        return None

    matches = sorted(candidate_dir.glob(f"*_sim_{sim_id:05d}.csv"))
    return matches[0] if matches else None


def load_path_from_candidate_csv(
    selection_dir: Path,
    sim_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = find_candidate_csv(selection_dir, sim_id)
    if path is None:
        return None

    df = pd.read_csv(path)
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
            f"Nie znaleziono trajektorii dla simulation_id={sim_id} "
            f"w folderze {selection_dir}."
        )

    lambda_grid, N_grid, dN = loaded
    return lambda_grid, N_grid, dN, source


def get_number(params: dict, key: str) -> float:
    if key not in params:
        raise KeyError(f"Brak parametru '{key}' w pliku parametrów.")
    return float(params[key])


def validate_common_parameters(
    insured_params: dict,
    uninsured_params: dict,
) -> None:
    keys = [
        "T",
        "N",
        "N_lambda",
        "N_h",
        "lambda_max",
        "H_max",
        "H_0",
        "rho",
        "v",
        "a_1",
        "zeta",
        "gamma",
        "z_max",
    ]

    mismatches = []
    for key in keys:
        insured_value = get_number(insured_params, key)
        uninsured_value = get_number(uninsured_params, key)

        if not np.isclose(
            insured_value,
            uninsured_value,
            rtol=1e-10,
            atol=1e-12,
        ):
            mismatches.append(
                f"{key}: ubezpieczenie={insured_value}, "
                f"bez ubezpieczenia={uninsured_value}"
            )

    if mismatches:
        raise ValueError(
            "Warianty nie mają zgodnych parametrów wspólnych:\n"
            + "\n".join(mismatches)
        )


def leakage_S1(
    H: np.ndarray | float,
    params: dict,
) -> np.ndarray | float:
    v = float(params["v"])
    a1 = float(params["a_1"])
    return v * np.exp(-a1 * H)


def premium(
    H: np.ndarray | float,
    params: dict,
) -> np.ndarray | float:
    v = float(params["v"])
    pi0 = float(params["pi_0"])
    r = float(params["r"])

    S = leakage_S1(H, params)
    return pi0 * (1.0 - r * (1.0 - S / v))


def investment_cost(
    z: np.ndarray | float,
    params: dict,
) -> np.ndarray | float:
    zeta = float(params["zeta"])
    gamma = float(params["gamma"])
    return zeta * z + 0.5 * gamma * z ** 2


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


def compute_variant_dynamics(
    *,
    group: str,
    simulation_id: int,
    variant: str,
    insured: bool,
    lambda_grid: np.ndarray,
    N_grid: np.ndarray,
    dN: np.ndarray,
    z_policy: np.ndarray,
    params: dict,
) -> tuple[pd.DataFrame, dict]:
    T = float(params["T"])
    N = len(lambda_grid) - 1
    dt = T / N

    H0 = float(params["H_0"])
    rho = float(params["rho"])
    v = float(params["v"])
    eta_retained = float(params["eta_d"])

    t_grid = np.linspace(0.0, T, N + 1)
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
            H_grid[n + 1] = (
                exp_rho * H_grid[n]
                + ((1.0 - exp_rho) / rho) * z_n
            )

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

    S_grid = leakage_S1(H_grid, params)
    reduction_grid = 1.0 - S_grid / v

    investment_rate = investment_cost(z_grid, params)

    if insured:
        premium_rate = np.asarray(premium(H_grid, params), dtype=float)
    else:
        premium_rate = np.zeros(N + 1, dtype=float)

    loss_rate = eta_retained * S_grid * lambda_grid
    risk_cost_rate = premium_rate + loss_rate
    total_running_cost_rate = investment_rate + risk_cost_rate

    cum_investment = np.zeros(N + 1, dtype=float)
    cum_premium = np.zeros(N + 1, dtype=float)
    cum_loss = np.zeros(N + 1, dtype=float)
    cum_risk_cost = np.zeros(N + 1, dtype=float)
    cum_total_running_cost = np.zeros(N + 1, dtype=float)

    cum_investment[1:] = np.cumsum(investment_rate[:-1] * dt)
    cum_premium[1:] = np.cumsum(premium_rate[:-1] * dt)
    cum_loss[1:] = np.cumsum(loss_rate[:-1] * dt)
    cum_risk_cost[1:] = np.cumsum(risk_cost_rate[:-1] * dt)
    cum_total_running_cost[1:] = np.cumsum(
        total_running_cost_rate[:-1] * dt
    )

    dN_full = np.zeros(N + 1, dtype=int)
    dN_full[:-1] = dN

    df = pd.DataFrame(
        {
            "group": group,
            "simulation_id": simulation_id,
            "variant": variant,
            "insured": insured,
            "t": t_grid,
            "lambda_t": lambda_grid,
            "N_t": N_grid,
            "dN_interval_to_next": dN_full,
            "z_star": z_grid,
            "H_t": H_grid,
            "S_H_v": S_grid,
            "reduction_1_minus_S_over_v": reduction_grid,
            "investment_cost_rate": investment_rate,
            "premium_rate": premium_rate,
            "loss_rate": loss_rate,
            "risk_cost_rate": risk_cost_rate,
            "total_running_cost_rate": total_running_cost_rate,
            "cum_investment_cost": cum_investment,
            "cum_premium": cum_premium,
            "cum_loss": cum_loss,
            "cum_risk_cost": cum_risk_cost,
            "cum_total_running_cost": cum_total_running_cost,
        }
    )

    metrics = {
        "group": group,
        "simulation_id": int(simulation_id),
        "variant": variant,
        "insured": bool(insured),
        "T": T,
        "N": int(N),
        "total_attacks": int(N_grid[-1]),
        "max_lambda": float(np.max(lambda_grid)),
        "integrated_lambda": float(np.sum(lambda_grid[:-1]) * dt),
        "eta_retained": eta_retained,
        "z_0": float(z_grid[0]),
        "max_z": float(np.max(z_grid)),
        "z_T": float(z_grid[-1]),
        "integrated_z": float(np.sum(z_grid[:-1]) * dt),
        "max_H": float(np.max(H_grid)),
        "H_T": float(H_grid[-1]),
        "reduction_T": float(reduction_grid[-1]),
        "cum_investment_cost_T": float(cum_investment[-1]),
        "cum_premium_T": float(cum_premium[-1]),
        "cum_loss_T": float(cum_loss[-1]),
        "cum_risk_cost_T": float(cum_risk_cost[-1]),
        "cum_total_running_cost_T": float(cum_total_running_cost[-1]),
        "lambda_clip_count": int(lambda_clip_count),
        "H_clip_count": int(H_clip_count),
    }

    return df, metrics


def format_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.8,
        alpha=0.55,
    )
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def rising_intensity_intervals(
    t: np.ndarray,
    lambda_t: np.ndarray,
) -> list[tuple[float, float]]:
    diff = np.diff(lambda_t)
    rising = diff > 1e-12

    intervals: list[tuple[float, float]] = []
    start: int | None = None

    for i, is_rising in enumerate(rising):
        if is_rising and start is None:
            start = i
        elif not is_rising and start is not None:
            intervals.append((float(t[start]), float(t[i])))
            start = None

    if start is not None:
        intervals.append((float(t[start]), float(t[-1])))

    return intervals


def add_intensity_shading(
    ax: plt.Axes,
    t: np.ndarray,
    lambda_t: np.ndarray,
) -> None:
    for left, right in rising_intensity_intervals(t, lambda_t):
        ax.axvspan(
            left,
            right,
            color=COLOR_SHADE,
            alpha=0.18,
            linewidth=0,
            zorder=0,
        )


def choose_top_legend_location(
    y_insured: np.ndarray,
    y_uninsured: np.ndarray,
) -> str:
    y1 = np.asarray(y_insured, dtype=float)
    y2 = np.asarray(y_uninsured, dtype=float)

    y_min = float(np.nanmin([np.nanmin(y1), np.nanmin(y2)]))
    y_max = float(np.nanmax([np.nanmax(y1), np.nanmax(y2)]))

    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return "upper right"

    y1_normalized = (y1 - y_min) / (y_max - y_min)
    y2_normalized = (y2 - y_min) / (y_max - y_min)

    quarter = max(1, len(y1_normalized) // 4)

    left_occupancy = float(
        np.nanmean(
            np.maximum(
                y1_normalized[:quarter],
                y2_normalized[:quarter],
            )
        )
    )
    right_occupancy = float(
        np.nanmean(
            np.maximum(
                y1_normalized[-quarter:],
                y2_normalized[-quarter:],
            )
        )
    )

    return (
        "upper right"
        if right_occupancy <= left_occupancy
        else "upper left"
    )


def padded_limits(
    arrays: list[np.ndarray],
    padding_share: float = 0.05,
) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(array, dtype=float).ravel() for array in arrays]
    )
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (0.0, 1.0)

    lower = min(0.0, float(np.min(values)))
    upper = max(0.0, float(np.max(values)))

    span = upper - lower
    if span <= 0.0:
        span = max(abs(upper), 1.0)

    upper += padding_share * span
    return (0.0, upper)


def shared_axis_limits(
    baseline_insured: pd.DataFrame,
    baseline_uninsured: pd.DataFrame,
    stress_insured: pd.DataFrame,
    stress_uninsured: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    frames = [
        baseline_insured,
        baseline_uninsured,
        stress_insured,
        stress_uninsured,
    ]

    return {
        "z_star": padded_limits(
            [frame["z_star"].to_numpy() for frame in frames]
        ),
        "H_t": padded_limits(
            [frame["H_t"].to_numpy() for frame in frames]
        ),
        "cum_risk_cost": padded_limits(
            [frame["cum_risk_cost"].to_numpy() for frame in frames]
        ),
    }


def save_comparison_plot(
    *,
    insured_df: pd.DataFrame,
    uninsured_df: pd.DataFrame,
    column: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    add_intensity_shading(
        ax,
        insured_df["t"].to_numpy(dtype=float),
        insured_df["lambda_t"].to_numpy(dtype=float),
    )

    insured_line, = ax.plot(
        insured_df["t"],
        insured_df[column],
        color=COLOR_INSURED,
        linestyle="-",
        linewidth=2.15,
        label="z ubezpieczeniem",
        zorder=3,
    )

    uninsured_line, = ax.plot(
        uninsured_df["t"],
        uninsured_df[column],
        color=COLOR_UNINSURED,
        linestyle="-",
        linewidth=2.15,
        label="bez ubezpieczenia",
        zorder=3,
    )

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    format_axes(ax)

    if ylim is not None:
        ax.set_ylim(*ylim)

    legend_loc = choose_top_legend_location(
        insured_df[column].to_numpy(dtype=float),
        uninsured_df[column].to_numpy(dtype=float),
    )
    ax.legend(
        handles=[insured_line, uninsured_line],
        labels=["z ubezpieczeniem", "bez ubezpieczenia"],
        loc=legend_loc,
        fontsize=12,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    *,
    insured_df: pd.DataFrame,
    uninsured_df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    y_limits: dict[str, tuple[float, float]] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_limits = y_limits or {}

    save_comparison_plot(
        insured_df=insured_df,
        uninsured_df=uninsured_df,
        column="z_star",
        ylabel=r"$z_t^*$",
        out_path=out_dir / f"{prefix}_z_star_insurance_comparison.png",
        ylim=y_limits.get("z_star"),
    )

    save_comparison_plot(
        insured_df=insured_df,
        uninsured_df=uninsured_df,
        column="H_t",
        ylabel=r"$H_t$",
        out_path=out_dir / f"{prefix}_H_t_insurance_comparison.png",
        ylim=y_limits.get("H_t"),
    )

    save_comparison_plot(
        insured_df=insured_df,
        uninsured_df=uninsured_df,
        column="cum_risk_cost",
        ylabel=(
            r"$\int_0^t "
            r"[\pi(H_s)+\bar{\eta}_d S(H_s,v)\lambda_s]\,ds$"
        ),
        out_path=(
            out_dir
            / f"{prefix}_cumulative_risk_cost_insurance_comparison.png"
        ),
        ylim=y_limits.get("cum_risk_cost"),
    )


def latex_number(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}".replace(".", ",")


def scenario_label(group: str) -> str:
    return {
        "baseline": "Reprezentatywny",
        "stress": "Stresowy",
    }.get(group, group)


def variant_label(insured: bool) -> str:
    return "Z ubezpieczeniem" if insured else "Bez ubezpieczenia"


def write_latex_table(
    metrics_df: pd.DataFrame,
    out_path: Path,
) -> None:
    ordered = metrics_df.copy()
    ordered["group_order"] = ordered["group"].map(
        {"baseline": 0, "stress": 1}
    )
    ordered["variant_order"] = ordered["insured"].map(
        {True: 0, False: 1}
    )
    ordered = ordered.sort_values(
        ["group_order", "variant_order"]
    )

    rows = []
    for _, row in ordered.iterrows():
        rows.append(
            " & ".join(
                [
                    scenario_label(str(row["group"])),
                    variant_label(bool(row["insured"])),
                    latex_number(row["z_0"]),
                    latex_number(row["integrated_z"]),
                    latex_number(row["H_T"]),
                    latex_number(row["reduction_T"]),
                    latex_number(row["cum_investment_cost_T"]),
                    latex_number(row["cum_premium_T"]),
                    latex_number(row["cum_loss_T"]),
                    latex_number(row["cum_total_running_cost_T"]),
                ]
            )
            + r" \\"
        )

    content = r"""\begin{table}[H]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrr}
\toprule
\textbf{Scenariusz}
& \textbf{Wariant}
& \(\boldsymbol{z_0^*}\)
& \(\boldsymbol{\int_0^T z_t^*\,dt}\)
& \(\boldsymbol{H_T}\)
& \(\boldsymbol{1-S(H_T,v)/v}\)
& \(\boldsymbol{\int_0^T c(z_t^*)\,dt}\)
& \(\boldsymbol{\int_0^T \pi(H_t)\,dt}\)
& \(\boldsymbol{\int_0^T L_t^{\mathrm{exp}}\,dt}\)
& \(\boldsymbol{\int_0^T \ell_t\,dt}\) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""

    out_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Porównanie trajektorii modelu S1 z ubezpieczeniem "
            "i bez ubezpieczenia."
        )
    )

    parser.add_argument(
        "--baseline-id",
        type=int,
        default=DEFAULT_BASELINE_ID,
    )
    parser.add_argument(
        "--stress-id",
        type=int,
        default=DEFAULT_STRESS_ID,
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=DEFAULT_SELECTION_DIR,
    )
    parser.add_argument(
        "--insured-policy-dir",
        type=Path,
        default=DEFAULT_INSURED_POLICY_DIR,
    )
    parser.add_argument(
        "--uninsured-policy-dir",
        type=Path,
        default=DEFAULT_UNINSURED_POLICY_DIR,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--independent-y",
        action="store_true",
        help=(
            "Użyj niezależnych zakresów osi y dla scenariusza "
            "reprezentatywnego i stresowego."
        ),
    )

    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()

    print("=== Porównanie wpływu ubezpieczenia na trajektorie ===")
    print(f"baseline_id          = {args.baseline_id}")
    print(f"stress_id            = {args.stress_id}")
    print(f"selection_dir        = {args.selection_dir}")
    print(f"insured_policy_dir   = {args.insured_policy_dir}")
    print(f"uninsured_policy_dir = {args.uninsured_policy_dir}")
    print(f"out_dir              = {args.out_dir}")

    insured_params = load_params(args.insured_policy_dir)
    uninsured_params = load_params(args.uninsured_policy_dir)
    validate_common_parameters(insured_params, uninsured_params)

    insured_policy = load_policy(args.insured_policy_dir)
    uninsured_policy = load_policy(args.uninsured_policy_dir)

    out_dir = args.out_dir
    data_dir = out_dir / "data"
    plots_dir = out_dir / "plots"

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        ("baseline", args.baseline_id),
        ("stress", args.stress_id),
    ]

    scenario_results: dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}
    all_metrics: list[dict] = []

    for group, simulation_id in selected:
        print(
            f"\nPrzetwarzanie: {group}, "
            f"simulation_id={simulation_id}"
        )

        lambda_grid, N_grid, dN, source = load_selected_path(
            args.selection_dir,
            simulation_id,
        )
        print(f"Źródło trajektorii: {source}")

        insured_df, insured_metrics = compute_variant_dynamics(
            group=group,
            simulation_id=simulation_id,
            variant="insured",
            insured=True,
            lambda_grid=lambda_grid,
            N_grid=N_grid,
            dN=dN,
            z_policy=insured_policy,
            params=insured_params,
        )

        uninsured_df, uninsured_metrics = compute_variant_dynamics(
            group=group,
            simulation_id=simulation_id,
            variant="uninsured",
            insured=False,
            lambda_grid=lambda_grid,
            N_grid=N_grid,
            dN=dN,
            z_policy=uninsured_policy,
            params=uninsured_params,
        )

        combined_df = pd.concat(
            [insured_df, uninsured_df],
            ignore_index=True,
        )

        csv_path = (
            data_dir
            / f"{group}_sim_{simulation_id:05d}_insurance_comparison.csv"
        )
        combined_df.to_csv(csv_path, index=False)
        print(f"Zapisano CSV: {csv_path}")

        scenario_results[group] = (insured_df, uninsured_df)
        all_metrics.extend([insured_metrics, uninsured_metrics])

    baseline_insured, baseline_uninsured = scenario_results["baseline"]
    stress_insured, stress_uninsured = scenario_results["stress"]

    if args.independent_y:
        common_y_limits = None
    else:
        common_y_limits = shared_axis_limits(
            baseline_insured=baseline_insured,
            baseline_uninsured=baseline_uninsured,
            stress_insured=stress_insured,
            stress_uninsured=stress_uninsured,
        )

    for group, simulation_id in selected:
        insured_df, uninsured_df = scenario_results[group]

        make_plots(
            insured_df=insured_df,
            uninsured_df=uninsured_df,
            out_dir=plots_dir / f"{group}_sim_{simulation_id:05d}",
            prefix=f"{group}_sim_{simulation_id:05d}",
            y_limits=common_y_limits,
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = out_dir / "insurance_impact_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    table_path = out_dir / "insurance_impact_summary_table.tex"
    write_latex_table(metrics_df, table_path)

    print("\n=== Metryki porównawcze ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(metrics_df)

    print(f"\nZapisano metryki: {metrics_path}")
    print(f"Zapisano tabelę LaTeX: {table_path}")
    print(f"Zapisano wykresy w: {plots_dir.resolve()}")
    print(f"Czas wykonania: {time.time() - start_time:.2f} s")
    print("\nGotowe.")


if __name__ == "__main__":
    main()
