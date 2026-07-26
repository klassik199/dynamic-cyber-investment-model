"""Compute a cost decomposition along the model-implied mean trajectory."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grids import build_grids
from functions import (
    S_exp,
    premium,
    investment_cost,
    terminal_utility,
)
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


def make_output_dir(path: str = "cost_decomposition_results") -> Path:
    output_dir = Path(path)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def bilinear_interpolate(
    array: np.ndarray,
    lambda_value: float,
    H_value: float,
    grids: dict,
) -> float:
    lambda_grid = grids["lambda"]
    H_grid = grids["H"]

    d_lambda = grids["d_lambda"]
    dH = grids["dH"]

    N_lambda = len(lambda_grid) - 1
    N_h = len(H_grid) - 1

    lambda_value = min(max(lambda_value, lambda_grid[0]), lambda_grid[-1])
    H_value = min(max(H_value, H_grid[0]), H_grid[-1])

    j = int(np.floor((lambda_value - lambda_grid[0]) / d_lambda))
    k = int(np.floor((H_value - H_grid[0]) / dH))

    j = min(max(j, 0), N_lambda - 1)
    k = min(max(k, 0), N_h - 1)

    lambda_left = lambda_grid[j]
    H_left = H_grid[k]

    theta_lambda = (lambda_value - lambda_left) / d_lambda
    theta_H = (H_value - H_left) / dH

    value = (
        (1.0 - theta_lambda) * (1.0 - theta_H) * array[j, k]
        + theta_lambda * (1.0 - theta_H) * array[j + 1, k]
        + (1.0 - theta_lambda) * theta_H * array[j, k + 1]
        + theta_lambda * theta_H * array[j + 1, k + 1]
    )

    return float(value)


def solve_policy_and_store(
    params: dict,
    grids: dict,
) -> tuple[np.ndarray, np.ndarray]:
    V_next = initialize_terminal_value(grids)
    z_start = initialize_strategy(grids)

    N = int(params["N"])
    N_lambda_points = len(grids["lambda"])
    N_h_points = len(grids["H"])

    z_policy = np.empty(
        (N, N_lambda_points, N_h_points),
        dtype=np.float32,
    )

    print("Rozwiązywanie HJB i zapis polityki z^n...")
    print(f"N = {N}, wymiar pojedynczej warstwy z = {(N_lambda_points, N_h_points)}")

    for step in range(N):
        current_n = N - step - 1
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

        z_policy[current_n, :, :] = z_current.astype(np.float32)

        if step == 0 or (step + 1) % 250 == 0:
            print(
                f"krok {step + 1:4d}/{N}, "
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
                f"Krok: {step + 1}, diff_V={diff_V}, "
                f"diff_z={diff_z}, max_abs_V={max_abs_V}"
            )

        V_next = V_current
        z_start = z_current

    V_0 = V_next

    return V_0, z_policy


def simulate_mean_trajectory(
    params: dict,
    grids: dict,
    z_policy: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    N = int(params["N"])
    dt = float(grids["dt"])

    lambda_0 = float(params["lambda_0"])
    H_0 = float(params["H_0"])

    alfa = float(params["alfa"])
    ksi = float(params["ksi"])
    beta = float(params["beta"])
    rho = float(params["rho"])

    v = float(params["v"])
    a_1 = float(params["a_1"])
    pi_0 = float(params["pi_0"])
    r = float(params["r"])
    eta_d = float(params["eta_d"])
    zeta = float(params["zeta"])
    gamma = float(params["gamma"])

    t = np.linspace(0.0, float(params["T"]), N + 1)

    lambda_path = np.zeros(N + 1)
    H_path = np.zeros(N + 1)

    z_path = np.zeros(N)
    S_path = np.zeros(N)
    S_over_v_path = np.zeros(N)

    investment_cost_path = np.zeros(N)
    premium_path = np.zeros(N)
    retained_loss_path = np.zeros(N)
    total_running_cost_path = np.zeros(N)

    lambda_path[0] = lambda_0
    H_path[0] = H_0

    kappa = ksi - beta

    for n in range(N):
        lambda_n = lambda_path[n]
        H_n = H_path[n]

        z_n = bilinear_interpolate(
            z_policy[n, :, :],
            lambda_n,
            H_n,
            grids,
        )

        S_n = S_exp(H_n, v, a_1)
        pi_n = premium(S_n, v, pi_0, r)
        c_n = investment_cost(z_n, zeta, gamma)
        retained_n = eta_d * S_n * lambda_n

        z_path[n] = z_n
        S_path[n] = S_n
        S_over_v_path[n] = S_n / v

        investment_cost_path[n] = c_n
        premium_path[n] = pi_n
        retained_loss_path[n] = retained_n
        total_running_cost_path[n] = c_n + pi_n + retained_n


        if abs(kappa) > 1e-12:
            lambda_inf = ksi * alfa / kappa
            lambda_path[n + 1] = (
                lambda_inf
                + (lambda_n - lambda_inf) * np.exp(-kappa * dt)
            )
        else:
            lambda_path[n + 1] = lambda_n + ksi * alfa * dt


        if rho > 0:
            H_path[n + 1] = (
                np.exp(-rho * dt) * H_n
                + (1.0 - np.exp(-rho * dt)) / rho * z_n
            )
        else:
            H_path[n + 1] = H_n + dt * z_n

    cumulative_investment = np.concatenate(
        [[0.0], np.cumsum(investment_cost_path * dt)]
    )
    cumulative_premium = np.concatenate(
        [[0.0], np.cumsum(premium_path * dt)]
    )
    cumulative_retained_loss = np.concatenate(
        [[0.0], np.cumsum(retained_loss_path * dt)]
    )
    cumulative_total = np.concatenate(
        [[0.0], np.cumsum(total_running_cost_path * dt)]
    )

    terminal_H = H_path[-1]
    terminal_utility_value = float(terminal_utility(terminal_H))

    total_investment = float(cumulative_investment[-1])
    total_premium = float(cumulative_premium[-1])
    total_retained_loss = float(cumulative_retained_loss[-1])
    total_running_cost = float(cumulative_total[-1])

    objective_proxy = total_running_cost - terminal_utility_value

    path_df = pd.DataFrame(
        {
            "t": t[:-1],
            "lambda": lambda_path[:-1],
            "H": H_path[:-1],
            "z": z_path,
            "S": S_path,
            "S_over_v": S_over_v_path,
            "risk_reduction": 1.0 - S_over_v_path,
            "investment_cost_rate": investment_cost_path,
            "premium_rate": premium_path,
            "retained_loss_rate": retained_loss_path,
            "total_running_cost_rate": total_running_cost_path,
        }
    )

    cumulative_df = pd.DataFrame(
        {
            "t": t,
            "lambda": lambda_path,
            "H": H_path,
            "cumulative_investment": cumulative_investment,
            "cumulative_premium": cumulative_premium,
            "cumulative_retained_loss": cumulative_retained_loss,
            "cumulative_total_running_cost": cumulative_total,
        }
    )

    summary_df = pd.DataFrame(
        [
            {
                "total_investment_cost": total_investment,
                "total_premium": total_premium,
                "total_retained_loss": total_retained_loss,
                "total_running_cost": total_running_cost,
                "terminal_H": float(terminal_H),
                "terminal_utility": terminal_utility_value,
                "objective_proxy": objective_proxy,
                "share_investment": total_investment / total_running_cost,
                "share_premium": total_premium / total_running_cost,
                "share_retained_loss": total_retained_loss / total_running_cost,
                "mean_z": float(np.mean(z_path)),
                "max_z": float(np.max(z_path)),
                "mean_lambda": float(np.mean(lambda_path[:-1])),
                "final_lambda": float(lambda_path[-1]),
            }
        ]
    )

    return path_df, cumulative_df, summary_df


def plot_state_and_control(
    path_df: pd.DataFrame,
    cumulative_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(cumulative_df["t"], cumulative_df["lambda"], linewidth=2)
    axes[0].set_ylabel(r"$\lambda_t$")
    axes[0].set_title("Średnia trajektoria stanu i strategii")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cumulative_df["t"], cumulative_df["H"], linewidth=2)
    axes[1].set_ylabel(r"$H_t$")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(path_df["t"], path_df["z"], linewidth=2)
    axes[2].set_ylabel(r"$z_t^*$")
    axes[2].set_xlabel(r"Czas $t$")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "state_and_control_path.png", dpi=300)
    fig.savefig(output_dir / "state_and_control_path.pdf")
    plt.close(fig)


def plot_instantaneous_costs(
    path_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(path_df["t"], path_df["investment_cost_rate"], linewidth=2, label=r"$c(z_t)$")
    ax.plot(path_df["t"], path_df["premium_rate"], linewidth=2, label=r"$\pi(H_t)$")
    ax.plot(
        path_df["t"],
        path_df["retained_loss_rate"],
        linewidth=2,
        label=r"$\eta_d S(H_t,v)\lambda_t$",
    )

    ax.set_title("Chwilowe składniki kosztu wzdłuż optymalnej trajektorii")
    ax.set_xlabel(r"Czas $t$")
    ax.set_ylabel("Chwilowy koszt")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "instantaneous_cost_components.png", dpi=300)
    fig.savefig(output_dir / "instantaneous_cost_components.pdf")
    plt.close(fig)


def plot_cumulative_costs(
    cumulative_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        cumulative_df["t"],
        cumulative_df["cumulative_investment"],
        linewidth=2,
        label=r"$\int c(z_t)\,dt$",
    )
    ax.plot(
        cumulative_df["t"],
        cumulative_df["cumulative_premium"],
        linewidth=2,
        label=r"$\int \pi(H_t)\,dt$",
    )
    ax.plot(
        cumulative_df["t"],
        cumulative_df["cumulative_retained_loss"],
        linewidth=2,
        label=r"$\int \eta_d S(H_t,v)\lambda_t\,dt$",
    )
    ax.plot(
        cumulative_df["t"],
        cumulative_df["cumulative_total_running_cost"],
        linewidth=2,
        linestyle="--",
        label="Suma",
    )

    ax.set_title("Skumulowane składniki kosztu")
    ax.set_xlabel(r"Czas $t$")
    ax.set_ylabel("Skumulowany koszt")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_cost_components.png", dpi=300)
    fig.savefig(output_dir / "cumulative_cost_components.pdf")
    plt.close(fig)


def plot_cost_shares(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    row = summary_df.iloc[0]

    labels = [
        "Koszt inwestycji",
        "Składka",
        "Strata własna",
    ]

    values = [
        row["total_investment_cost"],
        row["total_premium"],
        row["total_retained_loss"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(labels, values)
    ax.set_title("Dekompozycja łącznego kosztu bieżącego")
    ax.set_ylabel("Łączny koszt")
    ax.grid(True, axis="y", alpha=0.3)

    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_dir / "cost_decomposition_bar.png", dpi=300)
    fig.savefig(output_dir / "cost_decomposition_bar.pdf")
    plt.close(fig)


def main() -> None:
    params = load_params()


    params["N"] = 1000
    params["N_lambda"] = 100
    params["N_h"] = 100

    params["z_max"] = 100
    params["H_max"] = 100
    params["gamma"] = 0.3

    output_dir = make_output_dir()
    grids = build_grids(params)

    print("Test dekompozycji kosztu")
    print(f"N = {params['N']}, N_lambda = {params['N_lambda']}, N_h = {params['N_h']}")
    print(f"z_max = {params['z_max']}, H_max = {params['H_max']}, gamma = {params['gamma']}")
    print(f"pi_0 = {params['pi_0']}, r = {params['r']}, eta_d = {params['eta_d']}")

    V_0, z_policy = solve_policy_and_store(params, grids)

    V_initial_state = bilinear_interpolate(
        V_0,
        float(params["lambda_0"]),
        float(params["H_0"]),
        grids,
    )

    path_df, cumulative_df, summary_df = simulate_mean_trajectory(
        params=params,
        grids=grids,
        z_policy=z_policy,
    )

    summary_df["V_initial_state"] = V_initial_state

    path_df.to_csv(output_dir / "cost_decomposition_path.csv", index=False)
    cumulative_df.to_csv(output_dir / "cost_decomposition_cumulative.csv", index=False)
    summary_df.to_csv(output_dir / "cost_decomposition_summary.csv", index=False)

    np.save(output_dir / "z_policy_cost_decomposition.npy", z_policy)

    plot_state_and_control(path_df, cumulative_df, output_dir)
    plot_instantaneous_costs(path_df, output_dir)
    plot_cumulative_costs(cumulative_df, output_dir)
    plot_cost_shares(summary_df, output_dir)

    print("\nPodsumowanie dekompozycji kosztu:")
    print(summary_df.to_string(index=False))

    print("\nZapisano wyniki w folderze:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
