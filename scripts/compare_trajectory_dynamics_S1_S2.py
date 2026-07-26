"""Compare trajectory dynamics obtained for the leakage functions S1 and S2."""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BASELINE_ID = 6221
DEFAULT_STRESS_ID = 57457

DEFAULT_S1_DIR = Path("trajectory_dynamics_S1")
DEFAULT_S2_DIR = Path("trajectory_dynamics_S2")
DEFAULT_OUT_DIR = Path("trajectory_comparison_S1_S2")

FIGSIZE = (8.3, 5.25)
DPI = 300

COLORS = {
    "z_star": {
        "S1": "#0B2F5B",   # granat
        "S2": "#6F94B5",   # jasniejszy niebieski
    },
    "H_t": {
        "S1": "#3F7652",   # ciemna zielen
        "S2": "#86A991",   # jasniejsza zielen
    },
    "premium_rate": {
        "S1": "#C66A1E",   # ciemny pomarancz
        "S2": "#E6A46B",   # jasniejszy pomarancz
    },
    "retained_loss_rate": {
        "S1": "#7A1F2B",   # bordo
        "S2": "#B96570",   # jasniejsze bordo
    },
}

LINESTYLES = {
    "S1": "-",
    "S2": "-",
}

LINEWIDTHS = {
    "S1": 2.2,
    "S2": 2.2,
}

COLOR_GROWTH_SHADE = "#DFC0C5"
GROWTH_SHADE_ALPHA = 0.36
GROWTH_SMOOTH_WINDOW = 21
GROWTH_MAX_GAP = 0.012
GROWTH_MIN_DURATION = 0.008
GROWTH_MIN_NET_RISE = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Porownuje dynamike trajektorii dla funkcji S_1 i S_2 "
            "dla tej samej reprezentatywnej i stresowej sciezki atakow."
        )
    )

    parser.add_argument(
        "--baseline-id",
        type=int,
        default=DEFAULT_BASELINE_ID,
        help=f"simulation_id trajektorii reprezentatywnej (domyslnie {DEFAULT_BASELINE_ID})",
    )
    parser.add_argument(
        "--stress-id",
        type=int,
        default=DEFAULT_STRESS_ID,
        help=f"simulation_id trajektorii stresowej (domyslnie {DEFAULT_STRESS_ID})",
    )
    parser.add_argument(
        "--s1-dir",
        type=Path,
        default=DEFAULT_S1_DIR,
        help="Folder wynikowy trajectory_dynamics_S1",
    )
    parser.add_argument(
        "--s2-dir",
        type=Path,
        default=DEFAULT_S2_DIR,
        help="Folder wynikowy trajectory_dynamics_S2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Folder wynikowy porownania",
    )

    return parser.parse_args()


def trajectory_csv_path(root_dir: Path, group: str, sim_id: int) -> Path:
    return root_dir / "data" / f"{group}_sim_{sim_id:05d}_dynamics.csv"


def load_trajectory(root_dir: Path, group: str, sim_id: int, variant: str) -> pd.DataFrame:
    path = trajectory_csv_path(root_dir, group, sim_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku dla {variant}:\n{path}\n\n"
            "Najpierw uruchom odpowiedni skrypt "
            "compute_selected_trajectory_dynamics_S1/S2."
        )

    df = pd.read_csv(path)

    required = {
        "t",
        "lambda_t",
        "N_t",
        "H_t",
        "z_star",
        "reduction_1_minus_S_over_v",
        "investment_cost_rate",
        "premium_rate",
        "retained_loss_rate",
        "cum_investment_cost",
        "cum_premium",
        "cum_retained_loss",
        "cum_total_running_cost",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Plik {path} nie zawiera wymaganych kolumn: {sorted(missing)}"
        )

    df = df.copy()
    df["variant"] = variant
    return df


def check_compatible_paths(df_s1: pd.DataFrame, df_s2: pd.DataFrame, group: str) -> None:
    """Sprawdza, czy porownywane warianty wykorzystuja te sama sciezke atakow."""
    if len(df_s1) != len(df_s2):
        raise ValueError(f"Rozna liczba punktow siatki dla grupy {group}.")

    if not np.allclose(df_s1["t"], df_s2["t"], atol=1e-12):
        raise ValueError(f"Rozne siatki czasu dla grupy {group}.")

    if not np.allclose(df_s1["lambda_t"], df_s2["lambda_t"], atol=1e-5):
        raise ValueError(
            f"Trajektorie lambda_t dla S_1 i S_2 nie sa identyczne w grupie {group}. "
            "Porownanie powinno wykorzystywac te sama realizacje procesu atakow."
        )

    if not np.array_equal(
        df_s1["N_t"].to_numpy(dtype=int),
        df_s2["N_t"].to_numpy(dtype=int),
    ):
        raise ValueError(
            f"Trajektorie N_t dla S_1 i S_2 nie sa identyczne w grupie {group}."
        )


def find_lambda_growth_intervals(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Find broader periods of increasing attack intensity for figure shading."""
    t = df["t"].to_numpy(dtype=float)
    lam = df["lambda_t"].to_numpy(dtype=float)

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

        if (
            bounded_left
            and bounded_right
            and (gap_end - gap_start) <= max_gap_steps
        ):
            rising[gap_start:gap_end] = True

    intervals: list[tuple[float, float]] = []
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

        if (
            duration >= GROWTH_MIN_DURATION
            and net_rise >= GROWTH_MIN_NET_RISE
        ):
            intervals.append((float(t[left]), float(t[right])))

        i += 1

    return intervals


def add_growth_shading(ax, intervals: list[tuple[float, float]]) -> None:
    for left, right in intervals:
        ax.axvspan(
            left,
            right,
            facecolor=COLOR_GROWTH_SHADE,
            alpha=GROWTH_SHADE_ALPHA,
            edgecolor="none",
            zorder=0,
        )


def format_axes(ax):
    """Apply the common axis formatting used in the figures."""
    ax.xaxis.grid(False)
    ax.yaxis.grid(
        True,
        color="#E8E8E8",
        alpha=1.0,
        linewidth=0.6,
        zorder=1,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=14)


def choose_top_legend_location(
    y_s1: np.ndarray,
    y_s2: np.ndarray,
) -> str:
    """
    Umieszcza legendę zawsze u góry:
    po prawej, jeżeli w prawym górnym rogu jest więcej wolnego miejsca,
    w przeciwnym razie po lewej.
    """
    y1 = np.asarray(y_s1, dtype=float)
    y2 = np.asarray(y_s2, dtype=float)

    y_min = float(np.nanmin([np.nanmin(y1), np.nanmin(y2)]))
    y_max = float(np.nanmax([np.nanmax(y1), np.nanmax(y2)]))

    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return "upper right"

    y1n = (y1 - y_min) / (y_max - y_min)
    y2n = (y2 - y_min) / (y_max - y_min)

    n = len(y1n)
    quarter = max(1, n // 4)

    left_occupancy = float(
        np.nanmean(np.maximum(y1n[:quarter], y2n[:quarter]))
    )
    right_occupancy = float(
        np.nanmean(np.maximum(y1n[-quarter:], y2n[-quarter:]))
    )

    return "upper right" if right_occupancy <= left_occupancy else "upper left"


def plot_comparison(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    column: str,
    ylabel: str,
    out_path: Path,
    intervals: list[tuple[float, float]],
    legend_loc: str = "best",
):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    add_growth_shading(ax, intervals)

    line_s1, = ax.plot(
        df_s1["t"],
        df_s1[column],
        color=COLORS[column]["S1"],
        linestyle=LINESTYLES["S1"],
        linewidth=LINEWIDTHS["S1"],
        label=r"$S_1$",
        zorder=3,
    )

    line_s2, = ax.plot(
        df_s2["t"],
        df_s2[column],
        color=COLORS[column]["S2"],
        linestyle=LINESTYLES["S2"],
        linewidth=LINEWIDTHS["S2"],
        label=r"$S_2$",
        zorder=3,
    )

    ax.set_xlabel(r"$t$", fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)

    format_axes(ax)

    legend_loc = choose_top_legend_location(
        df_s1[column].to_numpy(dtype=float),
        df_s2[column].to_numpy(dtype=float),
    )

    ax.legend(
        handles=[line_s1, line_s2],
        labels=[r"$S_1$", r"$S_2$"],
        loc=legend_loc,
        fontsize=16,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_four_panel(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    group: str,
    out_path: Path,
    intervals: list[tuple[float, float]],
):
    """Dodatkowo tworzy jedna pomocnicza figure 2x2."""
    specs = [
        ("z_star", r"$z_t^*$", "upper right"),
        ("H_t", r"$H_t$", "lower right"),
        ("premium_rate", r"$\pi(H_t)$", "upper right"),
        (
            "retained_loss_rate",
            r"$\eta_d S(H_t,v)\lambda_t$",
            "upper right",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharex=True)
    axes = axes.ravel()

    for ax, (column, ylabel, legend_loc) in zip(axes, specs):
        add_growth_shading(ax, intervals)

        line_s1, = ax.plot(
            df_s1["t"],
            df_s1[column],
            color=COLORS[column]["S1"],
            linestyle=LINESTYLES["S1"],
            linewidth=2.0,
            label=r"$S_1$",
            zorder=3,
        )
        line_s2, = ax.plot(
            df_s2["t"],
            df_s2[column],
            color=COLORS[column]["S2"],
            linestyle=LINESTYLES["S2"],
            linewidth=2.0,
            label=r"$S_2$",
            zorder=3,
        )

        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel(r"$t$", fontsize=16)
        format_axes(ax)
        legend_loc = choose_top_legend_location(
            df_s1[column].to_numpy(dtype=float),
            df_s2[column].to_numpy(dtype=float),
        )

        ax.legend(
            handles=[line_s1, line_s2],
            labels=[r"$S_1$", r"$S_2$"],
            loc=legend_loc,
            fontsize=15,
            frameon=True,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def summarize_variant(
    df: pd.DataFrame,
    group: str,
    variant: str,
    simulation_id: int,
) -> dict:
    t = df["t"].to_numpy(dtype=float)

    if len(t) < 2:
        raise ValueError("Trajektoria ma zbyt malo punktow.")

    dt = float(np.median(np.diff(t)))

    return {
        "scenario": group,
        "variant": variant,
        "simulation_id": int(simulation_id),
        "total_attacks": int(df["N_t"].iloc[-1]),
        "max_lambda": float(df["lambda_t"].max()),
        "integrated_lambda": float(np.sum(df["lambda_t"].iloc[:-1]) * dt),
        "z_0": float(df["z_star"].iloc[0]),
        "max_z": float(df["z_star"].max()),
        "integrated_z": float(np.sum(df["z_star"].iloc[:-1]) * dt),
        "H_T": float(df["H_t"].iloc[-1]),
        "reduction_T": float(df["reduction_1_minus_S_over_v"].iloc[-1]),
        "cum_investment_cost_T": float(df["cum_investment_cost"].iloc[-1]),
        "cum_premium_T": float(df["cum_premium"].iloc[-1]),
        "cum_retained_loss_T": float(df["cum_retained_loss"].iloc[-1]),
        "cum_total_running_cost_T": float(
            df["cum_total_running_cost"].iloc[-1]
        ),
    }


def format_pl_number(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def create_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    scenario_labels = {
        "baseline": "reprezentatywna",
        "stress": "stresowa",
    }

    lines = [
        r"\begin{center}",
        r"\textbf{Tabela X. Porównanie wyników trajektorii dla funkcji \(S_1\) i \(S_2\)}",
        r"\vspace{0.5em}",
        "",
        r"\resizebox{\textwidth}{!}{%",
        r"\renewcommand{\arraystretch}{1.20}",
        r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|}",
        r"\hline",
        (
            r"\textbf{Trajektoria} & \textbf{Funkcja} "
            r"& \textbf{\(\int_0^T z_t^*\,dt\)} "
            r"& \textbf{\(H_T\)} "
            r"& \textbf{\(1-S(H_T,v)/v\)} "
            r"& \textbf{\(C_{\mathrm{inv}}\)} "
            r"& \textbf{\(C_{\mathrm{skł}}\)} "
            r"& \textbf{\(C_{\mathrm{str}}\)} "
            r"& \textbf{\(C_{\mathrm{łącznie}}\)} \\"
        ),
        r"\hline",
    ]

    ordered = summary.copy()
    ordered["scenario_order"] = ordered["scenario"].map(
        {"baseline": 0, "stress": 1}
    )
    ordered["variant_order"] = ordered["variant"].map({"S1": 0, "S2": 1})
    ordered = ordered.sort_values(["scenario_order", "variant_order"])

    previous_scenario = None

    for _, row in ordered.iterrows():
        scenario = row["scenario"]
        scenario_text = scenario_labels.get(scenario, scenario)
        function_text = r"\(S_1\)" if row["variant"] == "S1" else r"\(S_2\)"

        lines.append(
            f"{scenario_text} & {function_text} "
            f"& {format_pl_number(row['integrated_z'])} "
            f"& {format_pl_number(row['H_T'])} "
            f"& {format_pl_number(row['reduction_T'])} "
            f"& {format_pl_number(row['cum_investment_cost_T'])} "
            f"& {format_pl_number(row['cum_premium_T'])} "
            f"& {format_pl_number(row['cum_retained_loss_T'])} "
            f"& {format_pl_number(row['cum_total_running_cost_T'])} \\\\"
        )
        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{center}",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    plots_dir = args.out_dir / "plots"
    data_dir = args.out_dir / "data"

    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selections = [
        ("baseline", args.baseline_id),
        ("stress", args.stress_id),
    ]

    summary_rows = []

    print("\n=== Porownanie trajektorii S_1 i S_2 ===")
    print(f"Trajektoria reprezentatywna: simulation_id={args.baseline_id}")
    print(f"Trajektoria stresowa:        simulation_id={args.stress_id}")

    for group, sim_id in selections:
        print(f"\nPrzetwarzanie: {group}, simulation_id={sim_id}")

        df_s1 = load_trajectory(args.s1_dir, group, sim_id, "S1")
        df_s2 = load_trajectory(args.s2_dir, group, sim_id, "S2")

        check_compatible_paths(df_s1, df_s2, group)

        intervals = find_lambda_growth_intervals(df_s1)

        plot_specs = [
            (
                "z_star",
                r"$z_t^*$",
                "z_star",
                "upper right",
            ),
            (
                "H_t",
                r"$H_t$",
                "H_t",
                "lower right",
            ),
            (
                "premium_rate",
                r"$\pi(H_t)$",
                "premium",
                "upper right",
            ),
            (
                "retained_loss_rate",
                r"$\eta_d S(H_t,v)\lambda_t$",
                "retained_loss",
                "upper right",
            ),
        ]

        for column, ylabel, filename_part, legend_loc in plot_specs:
            out_path = (
                plots_dir
                / f"{group}_sim_{sim_id:05d}_{filename_part}_S1_S2.png"
            )

            plot_comparison(
                df_s1=df_s1,
                df_s2=df_s2,
                column=column,
                ylabel=ylabel,
                out_path=out_path,
                intervals=intervals,
                legend_loc=legend_loc,
            )

        plot_four_panel(
            df_s1=df_s1,
            df_s2=df_s2,
            group=group,
            out_path=(
                plots_dir
                / f"{group}_sim_{sim_id:05d}_comparison_4panel_S1_S2.png"
            ),
            intervals=intervals,
        )

        summary_rows.append(
            summarize_variant(df_s1, group, "S1", sim_id)
        )
        summary_rows.append(
            summarize_variant(df_s2, group, "S2", sim_id)
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = data_dir / "trajectory_comparison_S1_S2_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    latex_path = args.out_dir / "trajectory_comparison_S1_S2_table.tex"
    create_latex_table(summary_df, latex_path)

    print("\nZapisano:")
    print(f"  wykresy: {plots_dir.resolve()}")
    print(f"  podsumowanie CSV: {summary_csv.resolve()}")
    print(f"  tabela LaTeX: {latex_path.resolve()}")
    print("\nGotowe.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nBLAD: {exc}", file=sys.stderr)
        sys.exit(1)
