# -*- coding: utf-8 -*-
"""
run_convergence_N_light.py

Lekki i bezpieczny skrypt do badania zbieżności po liczbie kroków czasowych N.

Ten skrypt NIE uruchamia full_grid_solver_outputs.py i NIE tworzy pełnych wyników
typu surface/heatmap/z_policy_full_float32.npy. Liczy tylko to, co jest potrzebne
do tabeli zbieżności:

    N = 250, 500, 1000, 2000
    N_lambda = 100
    N_h = 100

Wyniki:
    convergence_N_light_results/convergence_N_table.tex
    convergence_N_light_results/convergence_N_summary_excel_pl.csv
    convergence_N_light_results/log_main.txt
    convergence_N_light_results/diagnostics_N_XXXX.csv

Uruchomienie:
    py run_convergence_N_light.py
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grids import build_grids
from solver import (
    initialize_terminal_value,
    initialize_strategy,
    solve_policy_iteration_step,
)


# =====================================================================
# KONFIGURACJA
# =====================================================================

PARAMS_PATH = Path("model_params.json")

N_VALUES = [250, 500, 1000, 2000]

FIXED_N_LAMBDA = 100
FIXED_N_H = 100

LEAKAGE_FUNCTION = "exp"

OUT_DIR = Path("convergence_N_light_results")

# Diagnostyka mniej więcej co 10% kroków.
PRINT_FRACTION = 0.10

# Parametry iteracji polityki.
MAX_POLICY_ITER = 20

# Progi bezpieczeństwa, takie same jakościowo jak w finalnym solverze.
MAX_DIFF_V_ALLOWED = 1e3
MAX_DIFF_Z_ALLOWED = 1.0
MAX_ABS_V_ALLOWED = 1e8


# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, log_file: Path | None = None) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_params(base_params: dict[str, Any], N: int) -> dict[str, Any]:
    params = deepcopy(base_params)

    params["N"] = int(N)
    params["N_lambda"] = int(FIXED_N_LAMBDA)
    params["N_h"] = int(FIXED_N_H)

    return params


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def fmt_float(x: Any, digits: int = 6) -> str:
    if x is None:
        return ""
    try:
        x_float = float(x)
    except Exception:
        return str(x)
    if math.isnan(x_float):
        return ""
    return f"{x_float:.{digits}f}"


def latex_number(x: Any, digits: int = 6) -> str:
    return fmt_float(x, digits)


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """
    CSV pod polskiego Excela:
    - separator kolumn: ;
    - separator dziesiętny: ,
    """
    headers = [
        "N",
        "dt",
        "N_lambda",
        "N_h",
        "stable",
        "last_step",
        "runtime_min",
        "lambda_0",
        "H_0",
        "j0",
        "k0",
        "V_initial_grid_point",
        "z_initial_grid_point",
        "mean_z_0",
        "min_z_0",
        "max_z_0",
        "share_z_max_0",
        "share_z_zero_0",
        "abs_diff_V_vs_ref",
        "abs_diff_z_vs_ref",
        "rel_diff_V_vs_ref",
        "rel_diff_z_vs_ref",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(";".join(headers) + "\n")
        for row in rows:
            values = []
            for h in headers:
                v = row.get(h, "")
                if isinstance(v, float):
                    values.append(str(v).replace(".", ","))
                else:
                    values.append(str(v))
            f.write(";".join(values) + "\n")


def write_latex_table(rows: list[dict[str, Any]], path: Path) -> None:
    ref_N = rows[-1]["N"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Zbieżność rozwiązania numerycznego względem liczby kroków czasowych \(N\) "
        r"dla siatki przestrzennej \(100\times 100\).}"
    )
    lines.append(r"\label{tab:convergence_N}")
    lines.append(r"\begin{tabular}{rrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\(N\) & \(\Delta t\) & "
        r"\(V(0,\underline{\lambda},\underline{h})\) & "
        r"\(z^*(0,\underline{\lambda},\underline{h})\) & "
        r"\(\overline{z^*}_0\) & "
        rf"\(|\Delta V|_{{N={ref_N}}}\) & "
        rf"\(|\Delta z^*|_{{N={ref_N}}}\) \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        lines.append(
            f"{int(row['N'])} & "
            f"{latex_number(row.get('dt'), 6)} & "
            f"{latex_number(row.get('V_initial_grid_point'), 6)} & "
            f"{latex_number(row.get('z_initial_grid_point'), 6)} & "
            f"{latex_number(row.get('mean_z_0'), 6)} & "
            f"{latex_number(row.get('abs_diff_V_vs_ref'), 6)} & "
            f"{latex_number(row.get('abs_diff_z_vs_ref'), 6)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def add_reference_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ostatni wariant N traktujemy jako referencyjny, czyli tutaj N = 2000.
    """
    ref = rows[-1]
    V_ref = float(ref["V_initial_grid_point"])
    z_ref = float(ref["z_initial_grid_point"])

    for row in rows:
        V = float(row["V_initial_grid_point"])
        z = float(row["z_initial_grid_point"])

        row["abs_diff_V_vs_ref"] = abs(V - V_ref)
        row["abs_diff_z_vs_ref"] = abs(z - z_ref)

        row["rel_diff_V_vs_ref"] = abs(V - V_ref) / abs(V_ref) if V_ref != 0 else math.nan
        row["rel_diff_z_vs_ref"] = abs(z - z_ref) / abs(z_ref) if z_ref != 0 else math.nan

    return rows


def run_single_N(N: int, base_params: dict[str, Any], log_file: Path) -> dict[str, Any]:
    params = prepare_params(base_params, N)
    grids = build_grids(params)

    t_grid = grids["t"]
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    z_max = float(params["z_max"])
    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    j0 = nearest_index(lambda_grid, lambda_0)
    k0 = nearest_index(H_grid, H_0)

    dt = float(grids["dt"])

    log("=" * 80, log_file)
    log(f"Start N={N}, N_lambda={FIXED_N_LAMBDA}, N_h={FIXED_N_H}", log_file)
    log(f"dt={dt}, d_lambda={grids['d_lambda']}, dH={grids['dH']}", log_file)
    log(f"Funkcja wycieku: {LEAKAGE_FUNCTION}", log_file)
    log(f"Indeks lambda: j0={j0}, lambda[j0]={lambda_grid[j0]}", log_file)
    log(f"Indeks H: k0={k0}, H[k0]={H_grid[k0]}", log_file)

    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    stable = True
    last_step = 0
    diagnostics_rows = []

    print_every = max(1, int(round(N * PRINT_FRACTION)))
    start_time = time.time()

    for step in range(N):
        current_n = N - step - 1
        current_t = current_n * dt

        V_current, z_current, iterations, diff_V, diff_z = solve_policy_iteration_step(
            V_next=V_next,
            z_start=z_start,
            grids=grids,
            params=params,
            leakage_function=LEAKAGE_FUNCTION,
            max_iter=MAX_POLICY_ITER,
            verbose=False,
        )

        last_step = step + 1

        max_abs_V = max(abs(float(V_current.min())), abs(float(V_current.max())))
        share_z_max = float(np.mean(z_current >= z_max - 1e-8))
        share_z_zero = float(np.mean(z_current <= 1e-8))

        if (
            iterations >= MAX_POLICY_ITER
            or not np.isfinite(V_current).all()
            or not np.isfinite(z_current).all()
            or diff_V > MAX_DIFF_V_ALLOWED
            or diff_z > MAX_DIFF_Z_ALLOWED
            or max_abs_V > MAX_ABS_V_ALLOWED
        ):
            stable = False
            log(
                "UWAGA: możliwa niestabilność. "
                f"krok={step + 1}, diff_V={diff_V:.2e}, diff_z={diff_z:.2e}, "
                f"max_abs_V={max_abs_V:.4e}",
                log_file,
            )
            break

        if step == 0 or (step + 1) % print_every == 0 or step == N - 1:
            elapsed_min = (time.time() - start_time) / 60.0
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
                "elapsed_min": elapsed_min,
            }
            diagnostics_rows.append(row)

            log(
                f"krok {step + 1:4d}/{N}, "
                f"t={current_t:.4f}, "
                f"iter={iterations:2d}, "
                f"diff_V={diff_V:.2e}, "
                f"diff_z={diff_z:.2e}, "
                f"min V={V_current.min():.4f}, "
                f"max V={V_current.max():.4f}, "
                f"mean z={z_current.mean():.4f}, "
                f"max z={z_current.max():.4f}, "
                f"z=z_max={100.0 * share_z_max:.2f}%, "
                f"z=0={100.0 * share_z_zero:.2f}%, "
                f"czas={elapsed_min:.2f} min",
                log_file,
            )

        V_next = V_current
        z_start = z_current

    runtime_min = (time.time() - start_time) / 60.0

    V_0 = V_next
    z_0 = z_start

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_path = OUT_DIR / f"diagnostics_N_{N:04d}.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False)

    summary = {
        "N": int(N),
        "dt": float(dt),
        "N_lambda": int(FIXED_N_LAMBDA),
        "N_h": int(FIXED_N_H),
        "stable": bool(stable),
        "last_step": int(last_step),
        "runtime_min": float(runtime_min),
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
        "diagnostics_file": str(diagnostics_path),
    }

    log(
        f"Koniec N={N}: "
        f"V0={summary['V_initial_grid_point']:.6f}, "
        f"z0={summary['z_initial_grid_point']:.6f}, "
        f"mean_z_0={summary['mean_z_0']:.6f}, "
        f"czas={runtime_min:.2f} min",
        log_file,
    )

    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / "log_main.txt"

    if log_file.exists():
        log_file.unlink()

    log("Start lekkiego badania zbieżności po N.", log_file)
    log(f"N_VALUES={N_VALUES}", log_file)
    log(f"FIXED_N_LAMBDA={FIXED_N_LAMBDA}, FIXED_N_H={FIXED_N_H}", log_file)

    base_params = load_params(PARAMS_PATH)

    rows: list[dict[str, Any]] = []

    for idx, N in enumerate(N_VALUES, start=1):
        log(f"GLOBALNY KROK {idx}/{len(N_VALUES)}: N={N}", log_file)
        row = run_single_N(N, base_params, log_file)
        rows.append(row)

    rows = add_reference_errors(rows)

    summary_csv = OUT_DIR / "convergence_N_summary_excel_pl.csv"
    table_tex = OUT_DIR / "convergence_N_table.tex"

    write_summary_csv(rows, summary_csv)
    write_latex_table(rows, table_tex)

    pd.DataFrame(rows).to_csv(OUT_DIR / "convergence_N_summary_raw.csv", index=False)

    log("=" * 80, log_file)
    log("Gotowe. Zapisano:", log_file)
    log(str(summary_csv), log_file)
    log(str(table_tex), log_file)
    log(str(OUT_DIR / "convergence_N_summary_raw.csv"), log_file)


if __name__ == "__main__":
    main()
