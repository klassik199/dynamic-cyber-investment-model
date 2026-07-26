"""Simulate Hawkes attack paths and save basic diagnostics."""

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


PARAMS_PATH = Path("model_params.json")
OUT_DIR = Path("trajectory_test_outputs")

M_TEST = 100000
SEED0 = 20260705


def load_params(path: Path) -> dict:
    """Wczytuje parametry z model_params.json."""
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku {path}. Uruchom skrypt w folderze modelu."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_to_hawkespylib_params(params: dict) -> tuple[float, float, float]:
    """
    Mapowanie parametrów z modelu:
        d lambda_t = xi(alpha - lambda_t) dt + beta dN_t

    na parametry HawkesPyLib:
        lambda(t) = mu + (eta/theta) sum exp(-(t-ti)/theta)

    Zatem:
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


def expected_count_linear_hawkes(alpha: float, xi: float, beta: float, lambda0: float, T: float) -> float:
    """
    Teoretyczna wartość E[N_T] dla modelu:
        d lambda_t = xi(alpha-lambda_t)dt + beta dN_t.

    Dla m(t)=E[lambda_t]:
        m'(t)=xi alpha - (xi-beta)m(t).
    """
    k = xi - beta

    if k <= 0:
        return np.nan

    lambda_bar = xi * alpha / k

    return lambda_bar * T + (lambda0 - lambda_bar) * (1.0 - np.exp(-k * T)) / k


def simulate_hawkes_counts(
    T: float,
    N: int,
    alpha: float,
    xi: float,
    beta: float,
    lambda0: float,
    seed: int
):
    """
    Symuluje jedną ścieżkę procesu N_t za pomocą HawkesPyLib,
    a następnie wyznacza przyrosty Delta N_n na siatce czasowej.

    Uwaga:
    HawkesPyLib generuje czasy skoków. Trajektorię lambda_t
    na siatce odtwarzamy później zgodnie z dyskretną dynamiką modelu.
    """
    params_tmp = {
        "alfa": alpha,
        "ksi": xi,
        "beta": beta
    }
    mu, eta, theta = model_to_hawkespylib_params(params_tmp)

    sim = ExpHawkesProcessSimulation(mu=mu, eta=eta, theta=theta)
    timestamps = np.asarray(sim.simulate(T=T, seed=int(seed)), dtype=float)

    t_grid = np.linspace(0.0, T, N + 1)
    dt = T / N


    dN, _ = np.histogram(timestamps, bins=t_grid)


    N_grid = np.zeros(N + 1, dtype=int)
    N_grid[1:] = np.cumsum(dN)


    lambda_grid = np.empty(N + 1, dtype=float)
    lambda_grid[0] = lambda0

    exp_xi = np.exp(-xi * dt)

    for n in range(N):
        lambda_grid[n + 1] = (
            exp_xi * lambda_grid[n]
            + alpha * (1.0 - exp_xi)
            + beta * exp_xi * dN[n]
        )

    return timestamps, t_grid, dN, N_grid, lambda_grid


def main():
    start = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    params = load_params(PARAMS_PATH)

    alpha = float(params["alfa"])
    xi = float(params["ksi"])
    beta = float(params["beta"])
    lambda0 = float(params["lambda_0"])
    T = float(params["T"])
    N = int(params["N"])

    mu, eta, theta = model_to_hawkespylib_params(params)

    print("\n=== Test symulacji procesu Hawkesa N_t ===")
    print(f"alpha  = {alpha}")
    print(f"xi     = {xi}")
    print(f"beta   = {beta}")
    print(f"lambda0= {lambda0}")
    print(f"T      = {T}")
    print(f"N      = {N}")
    print("\nMapowanie do HawkesPyLib:")
    print(f"mu     = {mu}")
    print(f"eta    = beta / xi = {eta:.6f}")
    print(f"theta  = 1 / xi = {theta:.6f}")

    if abs(lambda0 - alpha) > 1e-10:
        print("\nUWAGA:")
        print("lambda_0 różni się od alfa.")
        print("HawkesPyLib startuje z intensywnością bazową mu=alfa.")
        print("W tym skrypcie lambda_t na siatce jest odtwarzana od lambda_0 zgodnie z Twoją dynamiką.")

    expected_NT = expected_count_linear_hawkes(alpha, xi, beta, lambda0, T)
    print(f"\nTeoretyczne E[N_T] dla modelu liniowego: {expected_NT:.3f}")


    timestamps, t_grid, dN, N_grid, lambda_grid = simulate_hawkes_counts(
        T=T,
        N=N,
        alpha=alpha,
        xi=xi,
        beta=beta,
        lambda0=lambda0,
        seed=SEED0
    )

    one_path_df = pd.DataFrame({
        "t": t_grid,
        "N_t": N_grid,
        "lambda_t": lambda_grid
    })


    increments_df = pd.DataFrame({
        "t_left": t_grid[:-1],
        "t_right": t_grid[1:],
        "dN": dN
    })

    one_path_df.to_csv(OUT_DIR / "hawkes_test_one_path_grid.csv", index=False)
    increments_df.to_csv(OUT_DIR / "hawkes_test_one_path_increments.csv", index=False)

    pd.DataFrame({"jump_time": timestamps}).to_csv(
        OUT_DIR / "hawkes_test_one_path_jump_times.csv",
        index=False
    )

    print("\nJedna ścieżka kontrolna:")
    print(f"liczba ataków = {len(timestamps)}")
    print(f"pierwsze czasy skoków = {timestamps[:10]}")


    total_attacks = np.empty(M_TEST, dtype=int)
    max_lambda = np.empty(M_TEST, dtype=float)
    integrated_lambda = np.empty(M_TEST, dtype=float)

    for m in range(M_TEST):
        _, t_grid_m, dN_m, N_grid_m, lambda_grid_m = simulate_hawkes_counts(
            T=T,
            N=N,
            alpha=alpha,
            xi=xi,
            beta=beta,
            lambda0=lambda0,
            seed=SEED0 + m
        )

        total_attacks[m] = N_grid_m[-1]
        max_lambda[m] = np.max(lambda_grid_m)
        integrated_lambda[m] = np.sum(lambda_grid_m[:-1]) * (T / N)

    summary_df = pd.DataFrame({
        "simulation_id": np.arange(M_TEST),
        "total_attacks": total_attacks,
        "max_lambda": max_lambda,
        "integrated_lambda": integrated_lambda
    })

    summary_df.to_csv(OUT_DIR / "hawkes_test_summary_1000.csv", index=False)

    print("\nPodsumowanie 100000 symulacji:")
    print(f"średnia liczba ataków       = {np.mean(total_attacks):.3f}")
    print(f"mediana liczby ataków       = {np.median(total_attacks):.3f}")
    print(f"90 percentyl liczby ataków  = {np.quantile(total_attacks, 0.90):.3f}")
    print(f"95 percentyl liczby ataków  = {np.quantile(total_attacks, 0.95):.3f}")
    print(f"max liczba ataków           = {np.max(total_attacks)}")
    print(f"średnia max(lambda_t)       = {np.mean(max_lambda):.3f}")
    print(f"średnia int lambda_t dt     = {np.mean(integrated_lambda):.3f}")


    plt.figure(figsize=(9, 4.8))
    plt.step(t_grid, N_grid, where="post")
    plt.xlabel("t")
    plt.ylabel(r"$N_t$")
    plt.title("Testowa ścieżka procesu zliczającego")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hawkes_test_Nt_path.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(t_grid, lambda_grid)
    plt.xlabel("t")
    plt.ylabel(r"$\lambda_t$")
    plt.title("Odtworzona intensywność na siatce")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hawkes_test_lambda_path.png", dpi=300)
    plt.close()


    median_attacks = np.median(total_attacks)
    q95_attacks = np.quantile(total_attacks, 0.95)

    X_MAX_PLOT = 140


    bin_edges = np.arange(
        np.floor(total_attacks.min() / 2) * 2,
        X_MAX_PLOT + 2,
        2
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.hist(
        total_attacks,
        bins=bin_edges,
        color="#0B2F5B",
        edgecolor="#061A33",
        linewidth=0.6,
        alpha=0.78,
        zorder=2
    )


    ax.axvline(
        median_attacks,
        color="#E57373",
        linestyle="--",
        linewidth=2.2,
        label="mediana",
        zorder=5
    )

    ax.axvline(
        q95_attacks,
        color="#9B0000",
        linestyle="--",
        linewidth=2.4,
        label="95 percentyl",
        zorder=5
    )


    ax.set_xlim(left=0, right=X_MAX_PLOT)


    y_top = ax.get_ylim()[1]
    x_offset = 1.2

    ax.text(
        median_attacks + x_offset,
        y_top * 0.98,
        f"{median_attacks:.0f}",
        color="#E57373",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        zorder=6
    )

    ax.text(
        q95_attacks + x_offset,
        y_top * 0.98,
        f"{q95_attacks:.0f}",
        color="#9B0000",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        zorder=6
    )

    ax.set_xlabel("liczba ataków w horyzoncie T", fontsize=12)
    ax.set_ylabel("liczba symulacji", fontsize=12)

    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=True,
        fontsize=11,
        loc="upper right"
    )

    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "hawkes_test_total_attacks_hist.png", dpi=300)
    plt.close()

    elapsed = time.time() - start

    print(f"\nZapisano wyniki w folderze: {OUT_DIR.resolve()}")
    print(f"Czas wykonania: {elapsed:.2f} s")
    print("\nTest zakończony poprawnie.\n")


if __name__ == "__main__":
    main()
