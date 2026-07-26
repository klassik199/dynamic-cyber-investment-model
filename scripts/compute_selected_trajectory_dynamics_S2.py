# compute_selected_trajectory_dynamics_S2.py
# Dla wybranych simulation_id odczytuje trajektorie lambda_t i N_t,
# wyznacza z_t^*, H_t, 1-S_2(H_t,v)/v oraz dekompozycje kosztow.
# Wyniki zapisuje do CSV i tworzy pojedyncze wykresy diagnostyczne.

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Domyslne ustawienia - mozna zmieniac tutaj albo przez argumenty w konsoli
# ============================================================

DEFAULT_BASELINE_ID = 6221
DEFAULT_STRESS_ID = 57457

# Proces atakow nie zalezy od S_1/S_2, dlatego domyslnie korzystamy
# z tej samej selekcji trajektorii co dla S_1. Mozna to zmienic argumentem --selection-dir.
DEFAULT_SELECTION_DIR = Path("trajectory_selection_S1")
DEFAULT_POLICY_DIR = Path("full_grid_results_S2")
DEFAULT_POISSON_POLICY_DIR = Path("full_grid_results_Poisson_S2")
DEFAULT_OUT_DIR = Path("trajectory_dynamics_S2")

VARIANT_NAME = "S2"
LEAKAGE_FUNCTION = "S2_rational"


# ============================================================
# Wczytywanie danych
# ============================================================

def load_json_first_existing(paths: list[Path]) -> dict:
    """Wczytuje pierwszy istniejacy plik JSON z listy."""
    for path in paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                print(f"Wczytano parametry z: {path}")
                return json.load(f)

    raise FileNotFoundError(
        "Nie znaleziono zadnego pliku z parametrami:\n"
        + "\n".join(str(p) for p in paths)
    )


def load_params(policy_dir: Path) -> dict:
    """Wczytuje parametry modelu, preferujac parametry zapisane przy solverze."""
    possible_paths = [
        policy_dir / "full_solver.params.json",
        policy_dir / "full_solver_params.json",
        Path("model_params.json"),
    ]
    return load_json_first_existing(possible_paths)


def load_policy(policy_dir: Path) -> np.ndarray:
    """Wczytuje tablice polityki z_t^*."""
    path = policy_dir / "z_policy_full_float32.npy"
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku polityki: {path}")

    z_policy = np.load(path, mmap_mode="r")
    print(f"Wczytano polityke: {path}")
    print(f"Ksztalt polityki: {z_policy.shape}")
    return z_policy


def load_poisson_policy(policy_dir: Path, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Wczytuje polityke optymalna dla jednorodnego procesu Poissona.
    Polityka ma wymiar (N+1, N_h+1), poniewaz zalezy tylko od (t,H).
    """
    policy_path = policy_dir / "z_policy_full_float32.npy"
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono polityki Poissona: {policy_path}\n"
            "Najpierw uruchom skrypt solve_full_policy_poisson.py."
        )

    z_policy = np.load(policy_path, mmap_mode="r")

    if z_policy.ndim != 2:
        raise ValueError(
            f"Polityka Poissona powinna byc dwuwymiarowa (t,H), "
            f"a ma ksztalt {z_policy.shape}."
        )

    H_grid_path = policy_dir / "H_grid.npy"
    if H_grid_path.exists():
        H_grid = np.asarray(np.load(H_grid_path), dtype=float)
    else:
        H_max = float(params["H_max"])
        H_grid = np.linspace(0.0, H_max, z_policy.shape[1])

    if len(H_grid) != z_policy.shape[1]:
        raise ValueError(
            "Liczba punktow siatki H nie zgadza sie z drugim wymiarem "
            "polityki Poissona."
        )

    print(f"Wczytano polityke Poissona: {policy_path}")
    print(f"Ksztalt polityki Poissona: {z_policy.shape}")

    return z_policy, H_grid


def load_selection_summary(selection_dir: Path) -> pd.DataFrame | None:
    """Wczytuje podsumowanie selekcji, jesli istnieje."""
    path = selection_dir / "trajectory_selection_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_path_from_all_grids(selection_dir: Path, sim_id: int):
    """
    Wczytuje lambda_t i N_t z plikow all_lambda_grid/all_N_grid.
    Uzywa mmap, wiec nie laduje calej macierzy do pamieci.
    """
    lambda_path = selection_dir / "all_lambda_grid_float32.npy"
    N_path = selection_dir / "all_N_grid_uint16.npy"

    if not lambda_path.exists() or not N_path.exists():
        return None

    lambda_all = np.load(lambda_path, mmap_mode="r")
    N_all = np.load(N_path, mmap_mode="r")

    if sim_id < 0 or sim_id >= lambda_all.shape[0]:
        raise IndexError(
            f"simulation_id={sim_id} poza zakresem zapisanych symulacji "
            f"0,...,{lambda_all.shape[0]-1}."
        )

    lambda_grid = np.asarray(lambda_all[sim_id], dtype=float)
    N_grid = np.asarray(N_all[sim_id], dtype=int)
    dN = np.diff(N_grid).astype(int)

    return lambda_grid, N_grid, dN


def find_candidate_csv(selection_dir: Path, sim_id: int) -> Path | None:
    """Szuka pliku CSV kandydata po simulation_id."""
    pattern = f"*_sim_{sim_id:05d}.csv"
    candidate_dir = selection_dir / "candidate_trajectories"

    if not candidate_dir.exists():
        return None

    matches = sorted(candidate_dir.glob(pattern))
    if len(matches) == 0:
        return None

    return matches[0]


def load_path_from_candidate_csv(selection_dir: Path, sim_id: int):
    """Awaryjnie wczytuje trajektorie z CSV kandydata."""
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


def load_selected_path(selection_dir: Path, sim_id: int):
    """
    Wczytuje wybrana trajektorie.
    Najpierw probuje uzyc all_lambda_grid/all_N_grid, a potem CSV kandydata.
    """
    out = load_path_from_all_grids(selection_dir, sim_id)
    source = "all_grids"

    if out is None:
        out = load_path_from_candidate_csv(selection_dir, sim_id)
        source = "candidate_csv"

    if out is None:
        raise FileNotFoundError(
            f"Nie znaleziono trajektorii dla simulation_id={sim_id}.\n"
            f"Sprawdz folder: {selection_dir}\n"
            f"Oczekiwano plikow all_lambda_grid/all_N_grid albo CSV kandydata."
        )

    lambda_grid, N_grid, dN = out
    return lambda_grid, N_grid, dN, source


# ============================================================
# Model i interpolacja polityki
# ============================================================

def S2_leakage(H: np.ndarray | float, params: dict) -> np.ndarray | float:
    """Funkcja prawdopodobienstwa wycieku S_2(H,v)=v/(1+a_2 H)^b."""
    v = float(params["v"])
    a2 = float(params["a_2"])
    b = float(params["b"])
    return v / ((1.0 + a2 * H) ** b)


def premium(H: np.ndarray | float, params: dict) -> np.ndarray | float:
    """Skladka pi(H)=pi_0(1-r(1-S(H,v)/v))."""
    v = float(params["v"])
    pi0 = float(params["pi_0"])
    r = float(params["r"])
    S = S2_leakage(H, params)
    reduction = 1.0 - S / v
    return pi0 * (1.0 - r * reduction)


def investment_cost(z: np.ndarray | float, params: dict) -> np.ndarray | float:
    """Koszt inwestycji c(z)=zeta z + gamma/2 z^2."""
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
    """
    Odczytuje z^*(t_n, lambda, H) z tablicy polityki przez interpolacje biliniowa.

    Zwraca:
        z_value          - wartosc interpolowana,
        lambda_clipped   - czy lambda zostala obcieta do zakresu siatki,
        H_clipped        - czy H zostalo obciete do zakresu siatki.
    """
    n_t, n_lam, n_H = z_policy.shape

    n = int(np.clip(n, 0, n_t - 1))

    lambda_max = float(params["lambda_max"])
    H_max = float(params["H_max"])

    lambda_clipped = lambda_value < 0.0 or lambda_value > lambda_max
    H_clipped = H_value < 0.0 or H_value > H_max

    lam = float(np.clip(lambda_value, 0.0, lambda_max))
    H = float(np.clip(H_value, 0.0, H_max))

    d_lambda = lambda_max / (n_lam - 1)
    dH = H_max / (n_H - 1)

    x = lam / d_lambda
    y = H / dH

    j0 = int(np.floor(x))
    k0 = int(np.floor(y))

    if j0 >= n_lam - 1:
        j0 = n_lam - 2
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

    z0 = (1.0 - wx) * z00 + wx * z10
    z1 = (1.0 - wx) * z01 + wx * z11
    z = (1.0 - wy) * z0 + wy * z1

    return z, lambda_clipped, H_clipped


def compute_dynamics_for_path(
    sim_id: int,
    group_name: str,
    lambda_grid: np.ndarray,
    N_grid: np.ndarray,
    dN: np.ndarray,
    z_policy: np.ndarray,
    params: dict,
):
    """Liczy z_t^*, H_t, redukcje prawdopodobienstwa wycieku i koszty."""
    T = float(params["T"])
    N = len(lambda_grid) - 1
    dt = T / N

    H0 = float(params["H_0"])
    rho = float(params["rho"])
    eta_d = float(params["eta_d"])
    v = float(params["v"])

    t_grid = np.linspace(0.0, T, N + 1)

    H_grid = np.empty(N + 1, dtype=float)
    z_grid = np.empty(N + 1, dtype=float)

    H_grid[0] = H0

    lambda_clip_count = 0
    H_clip_count = 0

    exp_rho = np.exp(-rho * dt) if rho != 0.0 else None

    # Odczyt z i aktualizacja H na przedzialach [t_n,t_{n+1}]
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

    # Wartosc z w ostatnim punkcie tylko do wykresu
    z_T, lambda_clipped, H_clipped = bilinear_policy_value(
        z_policy=z_policy,
        n=N,
        lambda_value=float(lambda_grid[N]),
        H_value=float(H_grid[N]),
        params=params,
    )
    z_grid[N] = z_T
    lambda_clip_count += int(lambda_clipped)
    H_clip_count += int(H_clipped)

    S_grid = S2_leakage(H_grid, params)
    reduction_grid = 1.0 - S_grid / v

    investment_grid = investment_cost(z_grid, params)
    premium_grid = premium(H_grid, params)
    retained_loss_grid = eta_d * S_grid * lambda_grid
    total_running_grid = investment_grid + premium_grid + retained_loss_grid

    # Skumulowane koszty liczone metoda lewych prostokatow na [0,T]
    cum_investment = np.zeros(N + 1, dtype=float)
    cum_premium = np.zeros(N + 1, dtype=float)
    cum_retained_loss = np.zeros(N + 1, dtype=float)
    cum_total_running = np.zeros(N + 1, dtype=float)

    cum_investment[1:] = np.cumsum(investment_grid[:-1] * dt)
    cum_premium[1:] = np.cumsum(premium_grid[:-1] * dt)
    cum_retained_loss[1:] = np.cumsum(retained_loss_grid[:-1] * dt)
    cum_total_running[1:] = np.cumsum(total_running_grid[:-1] * dt)

    dN_full = np.zeros(N + 1, dtype=int)
    dN_full[:-1] = dN

    df = pd.DataFrame({
        "group": group_name,
        "simulation_id": sim_id,
        "t": t_grid,
        "lambda_t": lambda_grid,
        "N_t": N_grid,
        "dN_interval_to_next": dN_full,
        "H_t": H_grid,
        "z_star": z_grid,
        "S2_H_v": S_grid,
        "reduction_1_minus_S_over_v": reduction_grid,
        "investment_cost_rate": investment_grid,
        "premium_rate": premium_grid,
        "retained_loss_rate": retained_loss_grid,
        "total_running_cost_rate": total_running_grid,
        "cum_investment_cost": cum_investment,
        "cum_premium": cum_premium,
        "cum_retained_loss": cum_retained_loss,
        "cum_total_running_cost": cum_total_running,
    })

    metrics = {
        "group": group_name,
        "simulation_id": int(sim_id),
        "T": T,
        "N": int(N),
        "total_attacks": int(N_grid[-1]),
        "max_lambda": float(np.max(lambda_grid)),
        "integrated_lambda": float(np.sum(lambda_grid[:-1]) * dt),
        "max_H": float(np.max(H_grid)),
        "H_T": float(H_grid[-1]),
        "max_z": float(np.max(z_grid)),
        "z_0": float(z_grid[0]),
        "z_T": float(z_grid[-1]),
        "integrated_z": float(np.sum(z_grid[:-1]) * dt),
        "reduction_T": float(reduction_grid[-1]),
        "cum_investment_cost_T": float(cum_investment[-1]),
        "cum_premium_T": float(cum_premium[-1]),
        "cum_retained_loss_T": float(cum_retained_loss[-1]),
        "cum_total_running_cost_T": float(cum_total_running[-1]),
        "lambda_clip_count": int(lambda_clip_count),
        "H_clip_count": int(H_clip_count),
    }

    return df, metrics


# ============================================================
# Pelny model Poissona
# ============================================================

def expected_hawkes_count(params: dict) -> float:
    """Teoretyczna wartosc E[N_T] dla liniowego procesu Hawkesa."""
    alpha = float(params["alfa"])
    xi = float(params["ksi"])
    beta = float(params["beta"])
    lambda0 = float(params["lambda_0"])
    T = float(params["T"])

    k = xi - beta

    if k <= 0.0:
        raise ValueError("Do benchmarku Poissona wymagane jest xi > beta.")

    lambda_bar = xi * alpha / k

    return (
        lambda_bar * T
        + (lambda0 - lambda_bar)
        * (1.0 - np.exp(-k * T)) / k
    )


def poisson_policy_value(
    z_policy_poisson: np.ndarray,
    H_grid_poisson: np.ndarray,
    n: int,
    H_value: float,
) -> tuple[float, bool]:
    """Interpoluje liniowo polityke Poissona po zmiennej H."""
    n = int(np.clip(n, 0, z_policy_poisson.shape[0] - 1))

    H_min = float(H_grid_poisson[0])
    H_max = float(H_grid_poisson[-1])
    H_clipped = H_value < H_min or H_value > H_max
    H = float(np.clip(H_value, H_min, H_max))

    z_value = float(
        np.interp(
            H,
            H_grid_poisson,
            np.asarray(z_policy_poisson[n], dtype=float),
        )
    )

    return z_value, H_clipped


def compute_poisson_dynamics(
    z_policy_poisson: np.ndarray,
    H_grid_poisson: np.ndarray,
    params: dict,
):
    """
    Wyznacza deterministyczna trajektorie optymalna dla pelnego modelu
    jednorodnego procesu Poissona, rozwiazanego osobno na siatce (t,H).
    """
    T = float(params["T"])
    N = int(params["N"])

    if z_policy_poisson.shape[0] != N + 1:
        raise ValueError(
            f"Polityka Poissona ma {z_policy_poisson.shape[0]} krokow czasu, "
            f"a oczekiwano {N + 1}."
        )

    dt = T / N
    t_grid = np.linspace(0.0, T, N + 1)

    expected_NT = expected_hawkes_count(params)
    lambda_P = expected_NT / T

    H0 = float(params["H_0"])
    rho = float(params["rho"])
    eta_d = float(params["eta_d"])
    v = float(params["v"])

    H_grid = np.empty(N + 1, dtype=float)
    z_grid = np.empty(N + 1, dtype=float)
    H_grid[0] = H0

    H_clip_count = 0
    exp_rho = np.exp(-rho * dt) if rho != 0.0 else None

    for n in range(N):
        z_n, H_clipped = poisson_policy_value(
            z_policy_poisson=z_policy_poisson,
            H_grid_poisson=H_grid_poisson,
            n=n,
            H_value=float(H_grid[n]),
        )

        z_grid[n] = z_n
        H_clip_count += int(H_clipped)

        if rho == 0.0:
            H_grid[n + 1] = H_grid[n] + dt * z_n
        else:
            H_grid[n + 1] = (
                exp_rho * H_grid[n]
                + ((1.0 - exp_rho) / rho) * z_n
            )

    z_T, H_clipped = poisson_policy_value(
        z_policy_poisson=z_policy_poisson,
        H_grid_poisson=H_grid_poisson,
        n=N,
        H_value=float(H_grid[N]),
    )
    z_grid[N] = z_T
    H_clip_count += int(H_clipped)

    lambda_grid = np.full(N + 1, lambda_P, dtype=float)
    expected_N_grid = lambda_P * t_grid

    S_grid = S2_leakage(H_grid, params)
    reduction_grid = 1.0 - S_grid / v

    investment_grid = investment_cost(z_grid, params)
    premium_grid = premium(H_grid, params)
    retained_loss_grid = eta_d * S_grid * lambda_grid
    total_running_grid = investment_grid + premium_grid + retained_loss_grid

    cum_investment = np.zeros(N + 1, dtype=float)
    cum_premium = np.zeros(N + 1, dtype=float)
    cum_retained_loss = np.zeros(N + 1, dtype=float)
    cum_total_running = np.zeros(N + 1, dtype=float)

    cum_investment[1:] = np.cumsum(investment_grid[:-1] * dt)
    cum_premium[1:] = np.cumsum(premium_grid[:-1] * dt)
    cum_retained_loss[1:] = np.cumsum(retained_loss_grid[:-1] * dt)
    cum_total_running[1:] = np.cumsum(total_running_grid[:-1] * dt)

    df = pd.DataFrame({
        "group": "poisson",
        "simulation_id": -1,
        "t": t_grid,
        "lambda_t": lambda_grid,
        "N_t": expected_N_grid,
        "H_t": H_grid,
        "z_star": z_grid,
        "reduction_1_minus_S_over_v": reduction_grid,
        "investment_cost_rate": investment_grid,
        "premium_rate": premium_grid,
        "retained_loss_rate": retained_loss_grid,
        "total_running_cost_rate": total_running_grid,
        "cum_investment_cost": cum_investment,
        "cum_premium": cum_premium,
        "cum_retained_loss": cum_retained_loss,
        "cum_total_running_cost": cum_total_running,
    })

    metrics = {
        "group": "poisson",
        "process": "Poisson",
        "simulation_id": -1,
        "T": T,
        "N": int(N),
        "total_attacks": float(expected_NT),
        "max_lambda": float(lambda_P),
        "integrated_lambda": float(lambda_P * T),
        "max_H": float(np.max(H_grid)),
        "H_T": float(H_grid[-1]),
        "max_z": float(np.max(z_grid)),
        "z_0": float(z_grid[0]),
        "z_T": float(z_grid[-1]),
        "integrated_z": float(np.sum(z_grid[:-1]) * dt),
        "reduction_T": float(reduction_grid[-1]),
        "cum_investment_cost_T": float(cum_investment[-1]),
        "cum_premium_T": float(cum_premium[-1]),
        "cum_retained_loss_T": float(cum_retained_loss[-1]),
        "cum_total_running_cost_T": float(cum_total_running[-1]),
        "lambda_clip_count": 0,
        "H_clip_count": int(H_clip_count),
    }

    return df, metrics


# ============================================================
# Estetyka wykresow i zaznaczanie okresow wzrostu intensywnosci
# ============================================================

FIGSIZE = (8.3, 5.25)
DPI = 300

COLOR_BURGUNDY = "#7A1F2B"       # lambda_t, N_t, strata zatrzymana
COLOR_NAVY = "#0B2F5B"           # z_t^*, koszt inwestycji
COLOR_ORANGE = "#D9822B"         # skladka
COLOR_STATE_GREEN = "#3F7652"    # H_t i wzgledna redukcja ryzyka
COLOR_POISSON_LAMBDA = "#B77A88" # lambda^P
COLOR_POISSON_Z = "#78A9D1"      # z^{*,P}
COLOR_GROWTH_SHADE = "#DFC0C5"   # tlo okresow wzrostu lambda_t

GROWTH_SHADE_ALPHA = 0.36
GROWTH_SMOOTH_WINDOW = 21
GROWTH_MAX_GAP = 0.012
GROWTH_MIN_DURATION = 0.008
GROWTH_MIN_NET_RISE = 2.0


def find_lambda_growth_intervals(df: pd.DataFrame) -> list[tuple[float, float]]:
    """
    Wyznacza szersze okresy wzrostu lambda_t, a nie pojedyncze skoki.

    Najpierw lambda_t jest lekko wygladzana srednia ruchoma. Nastepnie
    laczone sa krotkie przerwy pomiedzy sasiednimi odcinkami wzrostowymi.
    Pozostaja tylko okresy o minimalnym czasie trwania i minimalnym
    przyroscie intensywnosci. Te same przedzialy sa nanoszone na wszystkie
    wykresy danej trajektorii.
    """
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

    # Laczenie bardzo krotkich spadkow wewnatrz jednego klastra wzrostowego.
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

        if duration >= GROWTH_MIN_DURATION and net_rise >= GROWTH_MIN_NET_RISE:
            intervals.append((float(t[left]), float(t[right])))

        i += 1

    return intervals


def add_growth_shading(ax, intervals: list[tuple[float, float]]) -> None:
    """Dodaje jasne pionowe prostokaty dla okresow wzrostu lambda_t."""
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
    """Ujednolicona estetyka osi: tylko bardzo jasna siatka pozioma."""
    ax.xaxis.grid(False)
    ax.yaxis.grid(
        True,
        color="#E8E8E8",
        alpha=1.0,
        linewidth=0.6,
        zorder=1,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)


def save_single_series_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    ylabel: str,
    out_path: Path,
    color: str,
    step: bool = False,
    ylim: tuple[float, float] | None = None,
):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    intervals = find_lambda_growth_intervals(df)
    add_growth_shading(ax, intervals)

    if step:
        ax.step(
            df[x], df[y], where="post", linewidth=1.9,
            color=color, zorder=3,
        )
    else:
        ax.plot(df[x], df[y], linewidth=1.9, color=color, zorder=3)

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    format_axes(ax)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def save_hawkes_poisson_plot(
    hawkes_df: pd.DataFrame,
    poisson_df: pd.DataFrame,
    y_column: str,
    ylabel: str,
    out_path: Path,
    hawkes_color: str,
    poisson_color: str,
    hawkes_label: str,
    poisson_label: str,
    annotate_poisson_level: bool = False,
    ylim: tuple[float, float] | None = None,
):
    """Rysuje na wspolnej osi trajektorie modelu Hawkesa i Poissona."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    intervals = find_lambda_growth_intervals(hawkes_df)
    add_growth_shading(ax, intervals)

    ax.plot(
        hawkes_df["t"],
        hawkes_df[y_column],
        linewidth=1.9,
        color=hawkes_color,
        linestyle="-",
        label=hawkes_label,
        zorder=3,
    )

    ax.plot(
        poisson_df["t"],
        poisson_df[y_column],
        linewidth=2.0,
        color=poisson_color,
        linestyle="--",
        label=poisson_label,
        zorder=4,
    )

    if annotate_poisson_level:
        level = float(poisson_df[y_column].iloc[0])
        label_value = f"{level:.1f}".replace(".", ",")
        x_position = float(poisson_df["t"].iloc[-1]) * 0.77

        ax.annotate(
            label_value,
            xy=(x_position, level),
            xytext=(0, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=poisson_color,
            zorder=5,
        )

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    format_axes(ax)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper right", frameon=True, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def save_cost_decomposition_plot(
    df: pd.DataFrame,
    out_path: Path,
    cumulative: bool = False,
    ylim: tuple[float, float] | None = None,
):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    intervals = find_lambda_growth_intervals(df)
    add_growth_shading(ax, intervals)

    if cumulative:
        columns = [
            ("cum_investment_cost", r"$c(z_t^*)$", COLOR_NAVY),
            ("cum_premium", r"$\pi(H_t)$", COLOR_ORANGE),
            ("cum_retained_loss", r"$\bar{\eta}_d S(H_t,v)\lambda_t$", COLOR_BURGUNDY),
        ]
        ylabel = "koszt skumulowany"
    else:
        columns = [
            ("investment_cost_rate", r"$c(z_t^*)$", COLOR_NAVY),
            ("premium_rate", r"$\pi(H_t)$", COLOR_ORANGE),
            ("retained_loss_rate", r"$\bar{\eta}_d S(H_t,v)\lambda_t$", COLOR_BURGUNDY),
        ]
        ylabel = "koszt chwilowy"

    for col, label, color in columns:
        ax.plot(
            df["t"], df[col], linewidth=1.8,
            label=label, color=color, zorder=3,
        )

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    format_axes(ax)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper right", frameon=True, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def make_plots_for_path(
    df: pd.DataFrame,
    poisson_df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    y_limits: dict[str, tuple[float, float]] | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_limits = y_limits or {}

    save_hawkes_poisson_plot(
        hawkes_df=df,
        poisson_df=poisson_df,
        y_column="lambda_t",
        ylabel=r"$\lambda_t$",
        hawkes_color=COLOR_BURGUNDY,
        poisson_color=COLOR_POISSON_LAMBDA,
        hawkes_label=r"$\lambda_t$",
        poisson_label=r"$\lambda_t^P$",
        annotate_poisson_level=False,
        ylim=y_limits.get("lambda_t"),
        out_path=out_dir / f"{prefix}_lambda_t.png",
    )

    save_single_series_plot(
        df=df,
        x="t",
        y="N_t",
        ylabel=r"$N_t$",
        color=COLOR_BURGUNDY,
        out_path=out_dir / f"{prefix}_N_t.png",
        step=True,
        ylim=y_limits.get("N_t"),
    )

    save_hawkes_poisson_plot(
        hawkes_df=df,
        poisson_df=poisson_df,
        y_column="z_star",
        ylabel=r"$z_t^*$",
        hawkes_color=COLOR_NAVY,
        poisson_color=COLOR_POISSON_Z,
        hawkes_label=r"$z_t^*$",
        poisson_label=r"$z_t^{*,P}$",
        ylim=y_limits.get("z_star"),
        out_path=out_dir / f"{prefix}_z_star.png",
    )

    save_single_series_plot(
        df=df,
        x="t",
        y="H_t",
        ylabel=r"$H_t$",
        color=COLOR_STATE_GREEN,
        ylim=y_limits.get("H_t"),
        out_path=out_dir / f"{prefix}_H_t.png",
    )

    save_single_series_plot(
        df=df,
        x="t",
        y="reduction_1_minus_S_over_v",
        ylabel=r"$1-S_2(H_t,v)/v$",
        color=COLOR_STATE_GREEN,
        ylim=y_limits.get("reduction"),
        out_path=out_dir / f"{prefix}_reduction_1_minus_S_over_v.png",
    )

    save_cost_decomposition_plot(
        df=df,
        out_path=out_dir / f"{prefix}_cost_decomposition_instant.png",
        cumulative=False,
        ylim=y_limits.get("cost_instant"),
    )

    save_cost_decomposition_plot(
        df=df,
        out_path=out_dir / f"{prefix}_cost_decomposition_cumulative.png",
        cumulative=True,
        ylim=y_limits.get("cost_cumulative"),
    )




def shared_axis_limits(
    baseline_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    poisson_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """
    Wyznacza identyczne zakresy osi y dla odpowiadających sobie wykresów
    trajektorii reprezentatywnej i stresowej.
    """
    def padded_limits(
        arrays: list[np.ndarray],
        include_zero: bool = True,
        padding_share: float = 0.05,
    ) -> tuple[float, float]:
        values = np.concatenate([
            np.asarray(arr, dtype=float).ravel()
            for arr in arrays
        ])
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return (0.0, 1.0)

        lower = float(np.min(values))
        upper = float(np.max(values))

        if include_zero:
            lower = min(0.0, lower)
            upper = max(0.0, upper)

        span = upper - lower
        if span <= 0:
            span = max(abs(upper), 1.0)

        padding = padding_share * span
        lower_limit = lower - padding
        upper_limit = upper + padding

        # Dla wielkości nieujemnych nie pokazujemy dużego obszaru poniżej zera.
        if include_zero and lower >= 0:
            lower_limit = 0.0

        return (lower_limit, upper_limit)

    instant_columns = [
        "investment_cost_rate",
        "premium_rate",
        "retained_loss_rate",
    ]
    cumulative_columns = [
        "cum_investment_cost",
        "cum_premium",
        "cum_retained_loss",
    ]

    return {
        "lambda_t": padded_limits(
            [
                baseline_df["lambda_t"].to_numpy(),
                stress_df["lambda_t"].to_numpy(),
                poisson_df["lambda_t"].to_numpy(),
            ]
        ),
        "N_t": padded_limits(
            [
                baseline_df["N_t"].to_numpy(),
                stress_df["N_t"].to_numpy(),
            ]
        ),
        "z_star": padded_limits(
            [
                baseline_df["z_star"].to_numpy(),
                stress_df["z_star"].to_numpy(),
                poisson_df["z_star"].to_numpy(),
            ]
        ),
        "H_t": padded_limits(
            [
                baseline_df["H_t"].to_numpy(),
                stress_df["H_t"].to_numpy(),
            ]
        ),
        "reduction": padded_limits(
            [
                baseline_df["reduction_1_minus_S_over_v"].to_numpy(),
                stress_df["reduction_1_minus_S_over_v"].to_numpy(),
            ]
        ),
        "cost_instant": padded_limits(
            [
                baseline_df[col].to_numpy()
                for col in instant_columns
            ]
            + [
                stress_df[col].to_numpy()
                for col in instant_columns
            ]
        ),
        "cost_cumulative": padded_limits(
            [
                baseline_df[col].to_numpy()
                for col in cumulative_columns
            ]
            + [
                stress_df[col].to_numpy()
                for col in cumulative_columns
            ]
        ),
    }


# ============================================================
# Tabela podsumowujaca w LaTeX-u
# ============================================================

def format_pl_number(value: float, decimals: int = 3) -> str:
    """Formatuje liczbe z przecinkiem dziesietnym do tabeli LaTeX."""
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}".replace(".", ",")


def create_latex_summary_table(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Tworzy tabele: Hawkes reprezentatywny, Hawkes stresowy i Poisson."""
    row_labels = {
        "baseline": "reprezentatywna",
        "stress": "stresowa",
        "poisson": "Poisson",
    }

    ordered = metrics_df.copy()
    ordered["row_order"] = ordered["group"].map({"baseline": 0, "stress": 1, "poisson": 2})
    ordered = ordered.sort_values("row_order")

    variant_tex = r"\(S_2\)"

    lines = [
        r"\begin{center}",
        (
            r"\textbf{Tabela X. Porównanie trajektorii modelu Hawkesa "
            r"z modelem Poissona dla funkcji " + variant_tex + r"}"
        ),
        r"\vspace{0.5em}",
        "",
        r"\resizebox{\textwidth}{!}{%",
        r"\renewcommand{\arraystretch}{1.20}",
        r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}",
        r"\hline",
        (
            r"\textbf{Wariant} "
            r"& \textbf{\(N_T\)} "
            r"& \textbf{\(\max_t \lambda_t\)} "
            r"& \textbf{\(\int_0^T \lambda_t\,dt\)} "
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

    for _, row in ordered.iterrows():
        group = str(row["group"])
        label = row_labels.get(group, group)

        if group == "poisson":
            attacks_text = format_pl_number(row["total_attacks"])
        else:
            attacks_text = str(int(round(float(row["total_attacks"]))))

        lines.append(
            f"{label} "
            f"& {attacks_text} "
            f"& {format_pl_number(row['max_lambda'])} "
            f"& {format_pl_number(row['integrated_lambda'])} "
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


# ============================================================
# Główna część
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wyznacza dynamike z,H i koszty dla dwoch wybranych trajektorii procesu Hawkesa."
    )
    parser.add_argument("--baseline-id", type=int, default=DEFAULT_BASELINE_ID)
    parser.add_argument("--stress-id", type=int, default=DEFAULT_STRESS_ID)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument(
        "--poisson-policy-dir",
        type=Path,
        default=DEFAULT_POISSON_POLICY_DIR,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    start = time.time()
    args = parse_args()

    out_dir = args.out_dir
    data_dir = out_dir / "data"
    plots_dir = out_dir / "plots"

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Dynamika wybranych trajektorii ===")
    print(f"baseline_id   = {args.baseline_id}")
    print(f"stress_id     = {args.stress_id}")
    print(f"selection_dir = {args.selection_dir}")
    print(f"policy_dir    = {args.policy_dir}")
    print(f"poisson_dir   = {args.poisson_policy_dir}")
    print(f"out_dir       = {out_dir}")

    params = load_params(args.policy_dir)
    z_policy = load_policy(args.policy_dir)
    z_policy_poisson, H_grid_poisson = load_poisson_policy(
        args.poisson_policy_dir,
        params,
    )
    poisson_df, poisson_metrics = compute_poisson_dynamics(
        z_policy_poisson=z_policy_poisson,
        H_grid_poisson=H_grid_poisson,
        params=params,
    )
    poisson_csv_path = data_dir / "poisson_benchmark_dynamics.csv"
    poisson_df.to_csv(poisson_csv_path, index=False)
    summary_df = load_selection_summary(args.selection_dir)

    selected = [
        ("baseline", args.baseline_id),
        ("stress", args.stress_id),
    ]

    all_metrics = []
    all_dfs = []

    for group_name, sim_id in selected:
        print(f"\nPrzetwarzanie: {group_name}, simulation_id={sim_id}")

        lambda_grid, N_grid, dN, source = load_selected_path(args.selection_dir, sim_id)
        print(f"Zrodlo trajektorii: {source}")

        df, metrics = compute_dynamics_for_path(
            sim_id=sim_id,
            group_name=group_name,
            lambda_grid=lambda_grid,
            N_grid=N_grid,
            dN=dN,
            z_policy=z_policy,
            params=params,
        )

        metrics["process"] = "Hawkes"

        # Dopisujemy informacje z pliku selekcji, jesli sa dostepne.
        if summary_df is not None and "simulation_id" in summary_df.columns:
            row = summary_df.loc[summary_df["simulation_id"] == sim_id]
            if len(row) == 1:
                metrics["seed"] = int(row.iloc[0].get("seed", -1))
                metrics["median_score"] = float(row.iloc[0].get("median_score", np.nan))
                metrics["selection_candidate_type"] = str(row.iloc[0].get("candidate_type", ""))

        csv_path = data_dir / f"{group_name}_sim_{sim_id:05d}_dynamics.csv"
        df.to_csv(csv_path, index=False)
        print(f"Zapisano CSV: {csv_path}")

        all_metrics.append(metrics)
        all_dfs.append(df)

    if len(all_dfs) != 2:
        raise RuntimeError("Oczekiwano dwóch trajektorii: reprezentatywnej i stresowej.")

    baseline_df, stress_df = all_dfs
    common_y_limits = shared_axis_limits(
        baseline_df=baseline_df,
        stress_df=stress_df,
        poisson_df=poisson_df,
    )

    for (group_name, sim_id), df in zip(selected, all_dfs):
        make_plots_for_path(
            df=df,
            poisson_df=poisson_df,
            out_dir=plots_dir / f"{group_name}_sim_{sim_id:05d}",
            prefix=f"{group_name}_sim_{sim_id:05d}",
            y_limits=common_y_limits,
        )

    all_metrics.append(poisson_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = out_dir / "selected_trajectory_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    latex_table_path = out_dir / "selected_trajectory_summary_table.tex"
    create_latex_summary_table(metrics_df, latex_table_path)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_path = data_dir / "selected_trajectories_combined.csv"
    combined_df.to_csv(combined_path, index=False)

    config = {
        "variant": VARIANT_NAME,
        "leakage_function": LEAKAGE_FUNCTION,
        "baseline_id": int(args.baseline_id),
        "stress_id": int(args.stress_id),
        "selection_dir": str(args.selection_dir),
        "policy_dir": str(args.policy_dir),
        "poisson_policy_dir": str(args.poisson_policy_dir),
        "lambda_P": float(poisson_df["lambda_t"].iloc[0]),
        "out_dir": str(out_dir),
    }

    with open(out_dir / "selected_trajectory_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n=== Metryki wybranych trajektorii ===")
    print(metrics_df.to_string(index=False))

    elapsed = time.time() - start
    print(f"\nZapisano wyniki w folderze: {out_dir.resolve()}")
    print(f"Zapisano trajektorie Poissona: {poisson_csv_path}")
    print(f"Zapisano tabele LaTeX: {latex_table_path}")
    print(f"Czas wykonania: {elapsed:.2f} s")
    print("\nGotowe. Pojedyncze wykresy sa w folderze plots/.\n")


if __name__ == "__main__":
    main()
