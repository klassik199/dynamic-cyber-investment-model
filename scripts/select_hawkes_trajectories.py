# select_hawkes_trajectories_S1.py
# Symulacja wielu trajektorii procesu Hawkesa, wybór kandydatów
# reprezentatywnych i stresowych oraz przygotowanie wykresów lambda_t i N_t.

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from HawkesPyLib.simulation import ExpHawkesProcessSimulation
except ImportError:
    print("Brakuje pakietu HawkesPyLib.")
    print("Zainstaluj go poleceniem:")
    print("    pip install HawkesPyLib")
    sys.exit(1)


# ============================================================
# Ustawienia
# ============================================================

PARAMS_PATH = Path("model_params.json")

OUT_DIR = Path("trajectory_selection_S1")
CANDIDATES_DIR = OUT_DIR / "candidate_trajectories"
PLOTS_DIR = OUT_DIR / "plots"
JUMP_TIMES_DIR = OUT_DIR / "jump_times"

M = 100000
SEED0 = 20260705

N_MEDIAN_CANDIDATES = 100
N_STRESS_CANDIDATES = 100

# Ścieżki stresowe wybieramy z okolicy 95 percentyla skumulowanej intensywności,
# ale tylko spośród ścieżek co najmniej tak stresowych jak 95 percentyl.
STRESS_QUANTILE = 0.95

# Jeśli True, skok w obrębie przedziału siatki jest dyskontowany dokładnie
# do prawego końca przedziału:
# beta * exp(-xi * (t_{n+1} - tau_i)).
#
# Jeśli False, używana jest uproszczona aktualizacja z Algorytmu 1:
# beta * exp(-xi * dt) * Delta N_n.
USE_EXACT_JUMP_DECAY = True

# Czy zapisywać wszystkie siatki lambda_t i N_t.
# Dla M=10000 pliki są umiarkowane: około kilkudziesięciu MB.
SAVE_ALL_GRIDS = True


# ============================================================
# Funkcje pomocnicze
# ============================================================

def load_params(path: Path) -> dict:
    """Wczytuje parametry z pliku JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku {path}. Uruchom skrypt w folderze modelu."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_to_hawkespylib_params(params: dict) -> tuple[float, float, float]:
    """
    Model:
        d lambda_t = xi(alpha - lambda_t) dt + beta dN_t

    HawkesPyLib:
        lambda(t) = mu + (eta/theta) sum exp(-(t-ti)/theta)

    Mapowanie:
        mu = alpha
        theta = 1 / xi
        eta = beta / xi
    """
    alpha = float(params["alfa"])
    xi = float(params["ksi"])
    beta = float(params["beta"])

    mu = alpha
    theta = 1.0 / xi
    eta = beta / xi

    if eta >= 1.0:
        raise ValueError(
            f"Proces jest niestabilny dla beta/xi = {eta:.4f} >= 1."
        )

    return mu, eta, theta


def simulate_one_hawkes_path(
    sim: ExpHawkesProcessSimulation,
    T: float,
    N: int,
    alpha: float,
    xi: float,
    beta: float,
    lambda0: float,
    seed: int,
    use_exact_jump_decay: bool = True,
):
    """
    Symuluje jedną ścieżkę procesu N_t i odtwarza lambda_t na siatce.

    Zwraca:
        timestamps    - dokładne czasy skoków,
        dN            - przyrosty N_{t_{n+1}} - N_{t_n},
        N_grid        - proces N_t na siatce t_0,...,t_N,
        lambda_grid   - intensywność lambda_t na siatce t_0,...,t_N.
    """
    timestamps = np.asarray(sim.simulate(T=T, seed=int(seed)), dtype=float)

    t_grid = np.linspace(0.0, T, N + 1)
    dt = T / N
    exp_xi = np.exp(-xi * dt)

    dN, _ = np.histogram(timestamps, bins=t_grid)

    N_grid = np.zeros(N + 1, dtype=np.uint16)
    N_grid[1:] = np.cumsum(dN).astype(np.uint16)

    if use_exact_jump_decay:
        jump_term = np.zeros(N, dtype=np.float64)

        if len(timestamps) > 0:
            # Indeks przedziału [t_n, t_{n+1}), do którego wpada skok.
            idx = np.searchsorted(t_grid, timestamps, side="right") - 1
            idx = np.clip(idx, 0, N - 1)

            # Wkład skoku w intensywność na prawym końcu przedziału.
            decay_to_right = np.exp(-xi * (t_grid[idx + 1] - timestamps))
            np.add.at(jump_term, idx, beta * decay_to_right)
    else:
        jump_term = beta * exp_xi * dN.astype(float)

    lambda_grid = np.empty(N + 1, dtype=np.float32)
    lambda_grid[0] = lambda0

    for n in range(N):
        lambda_grid[n + 1] = (
            exp_xi * lambda_grid[n]
            + alpha * (1.0 - exp_xi)
            + jump_term[n]
        )

    return timestamps, dN.astype(np.uint16), N_grid, lambda_grid


def compute_median_scores(lambda_all: np.ndarray, N_all: np.ndarray) -> np.ndarray:
    """
    Wynik bliskości do mediany przebiegów.
    Im mniejszy score, tym bardziej reprezentatywna ścieżka.
    """
    median_lambda = np.median(lambda_all, axis=0)
    median_N = np.median(N_all.astype(float), axis=0)

    scale_lambda = np.maximum(np.std(lambda_all, axis=0), 1.0)
    scale_N = np.maximum(np.std(N_all.astype(float), axis=0), 1.0)

    score_lambda = np.mean(((lambda_all - median_lambda) / scale_lambda) ** 2, axis=1)
    score_N = np.mean(((N_all.astype(float) - median_N) / scale_N) ** 2, axis=1)

    return score_lambda + 0.25 * score_N


def save_candidate_csv(
    out_path: Path,
    t_grid: np.ndarray,
    lambda_grid: np.ndarray,
    N_grid: np.ndarray,
    dN: np.ndarray,
):
    """
    Zapisuje trajektorię kandydata do CSV.
    Kolumna dN_interval_to_next oznacza przyrost w przedziale [t_n,t_{n+1}].
    Dla ostatniego wiersza przyjęto 0.
    """
    dN_full = np.zeros(len(t_grid), dtype=np.uint16)
    dN_full[:-1] = dN

    df = pd.DataFrame({
        "t": t_grid,
        "lambda_t": lambda_grid,
        "N_t": N_grid,
        "dN_interval_to_next": dN_full
    })

    df.to_csv(out_path, index=False)


def plot_overlay(
    t_grid: np.ndarray,
    selected_indices: np.ndarray,
    lambda_all: np.ndarray,
    N_all: np.ndarray,
    summary_df: pd.DataFrame,
    out_prefix: Path,
    max_lines: int | None = None,
):
    """
    Tworzy dwa wykresy nakładane: lambda_t i N_t.
    """
    if max_lines is not None:
        selected_indices = selected_indices[:max_lines]

    # Lambda
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for sim_id in selected_indices:
        row = summary_df.loc[summary_df["simulation_id"] == sim_id].iloc[0]
        label = (
            f"id={sim_id}, "
            f"N={int(row['total_attacks'])}, "
            f"I={row['integrated_lambda']:.1f}"
        )
        ax.plot(t_grid, lambda_all[sim_id], linewidth=1.1, alpha=0.85, label=label)

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(r"$\lambda_t$", fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)

    if len(selected_indices) <= 12:
        ax.legend(fontsize=8, frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_lambda_overlay.png"), dpi=300)
    plt.close()

    # N_t
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for sim_id in selected_indices:
        row = summary_df.loc[summary_df["simulation_id"] == sim_id].iloc[0]
        label = (
            f"id={sim_id}, "
            f"N={int(row['total_attacks'])}, "
            f"I={row['integrated_lambda']:.1f}"
        )
        ax.step(t_grid, N_all[sim_id], where="post", linewidth=1.1, alpha=0.85, label=label)

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(r"$N_t$", fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)

    if len(selected_indices) <= 12:
        ax.legend(fontsize=8, frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_Nt_overlay.png"), dpi=300)
    plt.close()


def plot_candidate_grid_pages(
    t_grid: np.ndarray,
    selected_indices: np.ndarray,
    lambda_all: np.ndarray,
    N_all: np.ndarray,
    summary_df: pd.DataFrame,
    out_prefix: Path,
    candidates_per_page: int = 10,
):
    """
    Tworzy strony z kandydatami.
    Każdy kandydat ma dwa wykresy: lambda_t oraz N_t.
    """
    n_pages = int(np.ceil(len(selected_indices) / candidates_per_page))

    for page in range(n_pages):
        page_indices = selected_indices[
            page * candidates_per_page:(page + 1) * candidates_per_page
        ]

        n_rows = len(page_indices)
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(12, max(2.0 * n_rows, 4.0)),
            sharex=True
        )

        if n_rows == 1:
            axes = np.array([axes])

        for row_pos, sim_id in enumerate(page_indices):
            row = summary_df.loc[summary_df["simulation_id"] == sim_id].iloc[0]

            label_text = (
                f"id={sim_id}, "
                f"N={int(row['total_attacks'])}, "
                f"max λ={row['max_lambda']:.1f}, "
                f"∫λdt={row['integrated_lambda']:.1f}"
            )

            ax_lam = axes[row_pos, 0]
            ax_N = axes[row_pos, 1]

            ax_lam.plot(t_grid, lambda_all[sim_id], linewidth=1.1)
            ax_lam.set_ylabel(r"$\lambda_t$", fontsize=9)
            ax_lam.grid(alpha=0.22, linewidth=0.5)
            ax_lam.text(
                0.01,
                0.92,
                label_text,
                transform=ax_lam.transAxes,
                ha="left",
                va="top",
                fontsize=8
            )

            ax_N.step(t_grid, N_all[sim_id], where="post", linewidth=1.1)
            ax_N.set_ylabel(r"$N_t$", fontsize=9)
            ax_N.grid(alpha=0.22, linewidth=0.5)

            ax_lam.tick_params(axis="both", labelsize=8)
            ax_N.tick_params(axis="both", labelsize=8)

        axes[-1, 0].set_xlabel("t", fontsize=10)
        axes[-1, 1].set_xlabel("t", fontsize=10)

        plt.tight_layout()

        suffix = f"_page_{page + 1:02d}.png"
        plt.savefig(out_prefix.with_name(out_prefix.name + suffix), dpi=300)
        plt.close()


# ============================================================
# Główna część
# ============================================================

def main():
    start = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    JUMP_TIMES_DIR.mkdir(parents=True, exist_ok=True)

    params = load_params(PARAMS_PATH)

    alpha = float(params["alfa"])
    xi = float(params["ksi"])
    beta = float(params["beta"])
    lambda0 = float(params["lambda_0"])
    T = float(params["T"])
    N = int(params["N"])
    lambda_max = float(params.get("lambda_max", np.inf))

    mu, eta, theta = model_to_hawkespylib_params(params)

    print("\n=== Wybór trajektorii procesu Hawkesa ===")
    print(f"M                      = {M}")
    print(f"T                      = {T}")
    print(f"N                      = {N}")
    print(f"alpha                  = {alpha}")
    print(f"xi                     = {xi}")
    print(f"beta                   = {beta}")
    print(f"lambda0                = {lambda0}")
    print(f"lambda_max             = {lambda_max}")
    print(f"USE_EXACT_JUMP_DECAY   = {USE_EXACT_JUMP_DECAY}")
    print("\nMapowanie do HawkesPyLib:")
    print(f"mu     = {mu}")
    print(f"eta    = {eta:.6f}")
    print(f"theta  = {theta:.6f}")

    t_grid = np.linspace(0.0, T, N + 1)

    lambda_all = np.empty((M, N + 1), dtype=np.float32)
    N_all = np.empty((M, N + 1), dtype=np.uint16)

    total_attacks = np.empty(M, dtype=np.uint16)
    max_lambda = np.empty(M, dtype=np.float32)
    integrated_lambda = np.empty(M, dtype=np.float32)

    # Dla późniejszego zapisu kandydatów przechowujemy tylko czasy skoków.
    all_timestamps = []

    sim = ExpHawkesProcessSimulation(mu=mu, eta=eta, theta=theta)

    print("\nSymulowanie trajektorii...")

    for m in range(M):
        timestamps, dN, N_grid, lambda_grid = simulate_one_hawkes_path(
            sim=sim,
            T=T,
            N=N,
            alpha=alpha,
            xi=xi,
            beta=beta,
            lambda0=lambda0,
            seed=SEED0 + m,
            use_exact_jump_decay=USE_EXACT_JUMP_DECAY
        )

        lambda_all[m] = lambda_grid
        N_all[m] = N_grid

        total_attacks[m] = N_grid[-1]
        max_lambda[m] = np.max(lambda_grid)
        integrated_lambda[m] = np.sum(lambda_grid[:-1]) * (T / N)

        all_timestamps.append(timestamps)

        if (m + 1) % 1000 == 0:
            print(f"  wykonano {m + 1}/{M}")

    summary_df = pd.DataFrame({
        "simulation_id": np.arange(M, dtype=int),
        "seed": SEED0 + np.arange(M, dtype=int),
        "total_attacks": total_attacks.astype(int),
        "max_lambda": max_lambda.astype(float),
        "integrated_lambda": integrated_lambda.astype(float),
        "exceeds_lambda_max": max_lambda > lambda_max
    })

    # --------------------------------------------------------
    # Wybór kandydatów medianowych
    # --------------------------------------------------------

    print("\nWybór kandydatów bliskich medianie...")

    median_score = compute_median_scores(lambda_all, N_all)
    summary_df["median_score"] = median_score

    median_candidate_ids = np.argsort(median_score)[:N_MEDIAN_CANDIDATES]

    # --------------------------------------------------------
    # Wybór kandydatów stresowych
    # --------------------------------------------------------

    print("Wybór kandydatów stresowych...")

    q95_integrated = np.quantile(integrated_lambda, STRESS_QUANTILE)

    stress_pool = np.where(integrated_lambda >= q95_integrated)[0]

    # Kandydaci najbliżsi progowi 95 percentyla od góry:
    # są stresowi, ale nie są ekstremalnymi wartościami odstającymi.
    stress_order = stress_pool[
        np.argsort(np.abs(integrated_lambda[stress_pool] - q95_integrated))
    ]

    stress_candidate_ids = stress_order[:N_STRESS_CANDIDATES]

    summary_df["candidate_type"] = ""
    summary_df.loc[median_candidate_ids, "candidate_type"] = "median_candidate"
    summary_df.loc[stress_candidate_ids, "candidate_type"] = "stress_candidate"

    # Jeżeli jakaś ścieżka znalazłaby się w obu grupach, dopisujemy oba oznaczenia.
    overlap = set(median_candidate_ids).intersection(set(stress_candidate_ids))
    for sim_id in overlap:
        summary_df.loc[summary_df["simulation_id"] == sim_id, "candidate_type"] = (
            "median_candidate;stress_candidate"
        )

    # --------------------------------------------------------
    # Zapis wyników
    # --------------------------------------------------------

    summary_path = OUT_DIR / "trajectory_selection_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    candidate_rows = []

    def save_group(group_name: str, candidate_ids: np.ndarray):
        for rank, sim_id in enumerate(candidate_ids, start=1):
            # Odtwarzamy dN z N_grid, żeby zapisać spójny plik CSV.
            N_grid = N_all[sim_id]
            dN = np.diff(N_grid).astype(np.uint16)

            csv_name = f"{group_name}_rank_{rank:03d}_sim_{sim_id:05d}.csv"
            jump_name = f"{group_name}_rank_{rank:03d}_sim_{sim_id:05d}_jump_times.csv"

            save_candidate_csv(
                out_path=CANDIDATES_DIR / csv_name,
                t_grid=t_grid,
                lambda_grid=lambda_all[sim_id],
                N_grid=N_grid,
                dN=dN
            )

            pd.DataFrame({"jump_time": all_timestamps[sim_id]}).to_csv(
                JUMP_TIMES_DIR / jump_name,
                index=False
            )

            row = summary_df.loc[summary_df["simulation_id"] == sim_id].iloc[0]

            candidate_rows.append({
                "group": group_name,
                "rank": rank,
                "simulation_id": int(sim_id),
                "seed": int(row["seed"]),
                "total_attacks": int(row["total_attacks"]),
                "max_lambda": float(row["max_lambda"]),
                "integrated_lambda": float(row["integrated_lambda"]),
                "median_score": float(row["median_score"]),
                "trajectory_csv": str(CANDIDATES_DIR / csv_name),
                "jump_times_csv": str(JUMP_TIMES_DIR / jump_name)
            })

    save_group("median", median_candidate_ids)
    save_group("stress", stress_candidate_ids)

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(OUT_DIR / "candidate_trajectories_index.csv", index=False)

    if SAVE_ALL_GRIDS:
        np.save(OUT_DIR / "all_lambda_grid_float32.npy", lambda_all)
        np.save(OUT_DIR / "all_N_grid_uint16.npy", N_all)

    # --------------------------------------------------------
    # Wykresy
    # --------------------------------------------------------

    print("Tworzenie wykresów kandydatów...")

    plot_overlay(
        t_grid=t_grid,
        selected_indices=median_candidate_ids,
        lambda_all=lambda_all,
        N_all=N_all,
        summary_df=summary_df,
        out_prefix=PLOTS_DIR / "median_candidates",
        max_lines=None
    )

    plot_candidate_grid_pages(
        t_grid=t_grid,
        selected_indices=median_candidate_ids,
        lambda_all=lambda_all,
        N_all=N_all,
        summary_df=summary_df,
        out_prefix=PLOTS_DIR / "median_candidates_grid",
        candidates_per_page=10
    )

    # Dla stresowych: jedna nakładka wszystkich 100 oraz strony po 10.
    plot_overlay(
        t_grid=t_grid,
        selected_indices=stress_candidate_ids,
        lambda_all=lambda_all,
        N_all=N_all,
        summary_df=summary_df,
        out_prefix=PLOTS_DIR / "stress_candidates_all100",
        max_lines=None
    )

    plot_candidate_grid_pages(
        t_grid=t_grid,
        selected_indices=stress_candidate_ids,
        lambda_all=lambda_all,
        N_all=N_all,
        summary_df=summary_df,
        out_prefix=PLOTS_DIR / "stress_candidates_grid",
        candidates_per_page=10
    )

    # --------------------------------------------------------
    # Podsumowanie w konsoli
    # --------------------------------------------------------

    print("\n=== Podsumowanie symulacji ===")
    print(f"średnia liczba ataków       = {np.mean(total_attacks):.3f}")
    print(f"mediana liczby ataków       = {np.median(total_attacks):.3f}")
    print(f"90 percentyl liczby ataków  = {np.quantile(total_attacks, 0.90):.3f}")
    print(f"95 percentyl liczby ataków  = {np.quantile(total_attacks, 0.95):.3f}")
    print(f"max liczba ataków           = {np.max(total_attacks)}")
    print(f"średnia max(lambda_t)       = {np.mean(max_lambda):.3f}")
    print(f"95 percentyl int lambda dt  = {q95_integrated:.3f}")
    print(f"liczba ścieżek > lambda_max = {int(np.sum(max_lambda > lambda_max))}")

    print("\nKandydaci medianowi:")
    print(
        candidate_df[candidate_df["group"] == "median"][
            ["rank", "simulation_id", "total_attacks", "max_lambda", "integrated_lambda", "median_score"]
        ].to_string(index=False)
    )

    print("\nPierwszych 20 kandydatów stresowych:")
    print(
        candidate_df[candidate_df["group"] == "stress"][
            ["rank", "simulation_id", "total_attacks", "max_lambda", "integrated_lambda", "median_score"]
        ].head(20).to_string(index=False)
    )

    elapsed = time.time() - start

    print(f"\nZapisano wyniki w folderze: {OUT_DIR.resolve()}")
    print(f"Czas wykonania: {elapsed:.2f} s")
    print("\nGotowe. Obejrzyj wykresy w folderze:")
    print(f"  {PLOTS_DIR.resolve()}")
    print("\nPo wyborze ścieżek podasz mi ich simulation_id albo nazwy plików CSV.\n")


if __name__ == "__main__":
    main()
