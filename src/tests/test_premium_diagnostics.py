import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import S_exp, S_rational, premium


def load_params(path: str = "model_params.json") -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)

    return params


def make_output_dir(path: str = "premium_diagnostics_results") -> Path:
    output_dir = Path(path)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def main() -> None:
    params = load_params()

    # Aktualne parametry bazowe do testów.
    params["z_max"] = 100
    params["H_max"] = 100
    params["gamma"] = 0.3

    output_dir = make_output_dir()

    H_max = float(params["H_max"])
    H_grid = np.linspace(0.0, H_max, 501)

    v = float(params["v"])
    a_1 = float(params["a_1"])
    a_2 = float(params["a_2"])
    b = float(params["b"])
    pi_0 = float(params["pi_0"])
    r = float(params["r"])

    # Bazowo używamy S_1, czyli funkcji wykładniczej.
    S1 = S_exp(H_grid, v, a_1)
    S2 = S_rational(H_grid, v, a_2, b)

    relative_risk_S1 = S1 / v
    relative_risk_S2 = S2 / v

    risk_reduction_S1 = 1.0 - relative_risk_S1
    risk_reduction_S2 = 1.0 - relative_risk_S2

    premium_S1 = premium(S1, v, pi_0, r)
    premium_S2 = premium(S2, v, pi_0, r)

    df = pd.DataFrame(
        {
            "H": H_grid,
            "S1": S1,
            "S2": S2,
            "S1_over_v": relative_risk_S1,
            "S2_over_v": relative_risk_S2,
            "risk_reduction_S1": risk_reduction_S1,
            "risk_reduction_S2": risk_reduction_S2,
            "premium_S1": premium_S1,
            "premium_S2": premium_S2,
        }
    )

    csv_path = output_dir / "premium_diagnostics.csv"
    df.to_csv(csv_path, index=False)

    print("Diagnostyka składki i względnej redukcji ryzyka")
    print(f"Zapisano dane do: {csv_path.resolve()}")

    selected_H = [0, 10, 20, 30, 50, 75, 100]

    print("\nWybrane wartości dla funkcji S_1:")
    print(
        f"{'H':>8s} "
        f"{'S(H,v)':>12s} "
        f"{'S/v':>12s} "
        f"{'1-S/v':>12s} "
        f"{'pi(H)':>12s}"
    )

    for H_value in selected_H:
        idx = nearest_index(H_grid, H_value)

        print(
            f"{H_grid[idx]:8.2f} "
            f"{S1[idx]:12.6f} "
            f"{relative_risk_S1[idx]:12.6f} "
            f"{risk_reduction_S1[idx]:12.6f} "
            f"{premium_S1[idx]:12.6f}"
        )

    # Wykres 1: względne ryzyko S/v.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(H_grid, relative_risk_S1, linewidth=2, label=r"$S_1(H,v)/v$")
    ax.plot(H_grid, relative_risk_S2, linewidth=2, linestyle="--", label=r"$S_2(H,v)/v$")

    ax.set_title(r"Względne prawdopodobieństwo wycieku $S(H,v)/v$")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Względne ryzyko $S(H,v)/v$")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "relative_risk_S_over_v.png", dpi=300)
    fig.savefig(output_dir / "relative_risk_S_over_v.pdf")
    plt.close(fig)

    # Wykres 2: względna redukcja ryzyka.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(H_grid, risk_reduction_S1, linewidth=2, label=r"$1-S_1(H,v)/v$")
    ax.plot(H_grid, risk_reduction_S2, linewidth=2, linestyle="--", label=r"$1-S_2(H,v)/v$")

    ax.set_title(r"Względna redukcja ryzyka")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Względna redukcja ryzyka")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "relative_risk_reduction.png", dpi=300)
    fig.savefig(output_dir / "relative_risk_reduction.pdf")
    plt.close(fig)

    # Wykres 3: składka.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(H_grid, premium_S1, linewidth=2, label=r"$\pi(H)$ dla $S_1$")
    ax.plot(H_grid, premium_S2, linewidth=2, linestyle="--", label=r"$\pi(H)$ dla $S_2$")

    ax.set_title(r"Składka ubezpieczeniowa $\pi(H;d,v,r)$")
    ax.set_xlabel(r"Poziom zabezpieczeń $H$")
    ax.set_ylabel(r"Składka")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "premium_function.png", dpi=300)
    fig.savefig(output_dir / "premium_function.pdf")
    plt.close(fig)

    # Jeżeli istnieją snapshoty strategii, porównujemy z(t, lambda_0, H) ze składką.
    snapshots_path = Path("policy_slices_results") / "policy_snapshots.npz"

    if snapshots_path.exists():
        data = np.load(snapshots_path)

        lambda_grid = data["lambda_grid"]
        H_policy_grid = data["H_grid"]

        lambda_0 = float(params["lambda_0"])
        j0 = nearest_index(lambda_grid, lambda_0)

        S_policy = S_exp(H_policy_grid, v, a_1)
        premium_policy = premium(S_policy, v, pi_0, r)
        relative_risk_policy = S_policy / v

        snapshot_keys = sorted(
            [key for key in data.files if key.startswith("z_t_")],
            key=lambda x: float(x.replace("z_t_", "").replace("_", "."))
        )

        policy_df = pd.DataFrame(
            {
                "H": H_policy_grid,
                "premium": premium_policy,
                "S_over_v": relative_risk_policy,
                "risk_reduction": 1.0 - relative_risk_policy,
            }
        )

        for key in snapshot_keys:
            t_value = float(key.replace("z_t_", "").replace("_", "."))
            z_snapshot = data[key]
            policy_df[f"z_t_{t_value:.2f}"] = z_snapshot[j0, :]

        policy_csv_path = output_dir / "policy_and_premium_H_slice.csv"
        policy_df.to_csv(policy_csv_path, index=False)

        print(f"\nZapisano przekrój strategii i składki do: {policy_csv_path.resolve()}")

        # Wspólny wykres: strategia i składka jako funkcje H.
        fig, ax1 = plt.subplots(figsize=(9, 5))

        for key in snapshot_keys:
            t_value = float(key.replace("z_t_", "").replace("_", "."))
            z_snapshot = data[key]
            ax1.plot(
                H_policy_grid,
                z_snapshot[j0, :],
                linewidth=2,
                label=fr"$z^*(t={t_value:.2f},\lambda_0,H)$",
            )

        ax1.set_xlabel(r"Poziom zabezpieczeń $H$")
        ax1.set_ylabel(r"Optymalna intensywność inwestycji $z^*$")
        ax1.set_ylim(0, params["z_max"] * 1.05)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(
            H_policy_grid,
            premium_policy,
            linewidth=2,
            linestyle=":",
            color="black",
            label=r"$\pi(H)$",
        )
        ax2.set_ylabel(r"Składka $\pi(H)$")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper right",
            fontsize=8,
        )

        ax1.set_title(r"Strategia $z^*(t,\lambda_0,H)$ oraz składka $\pi(H)$")

        fig.tight_layout()
        fig.savefig(output_dir / "policy_and_premium_H_slice.png", dpi=300)
        fig.savefig(output_dir / "policy_and_premium_H_slice.pdf")
        plt.close(fig)

    else:
        print(
            "\nNie znaleziono pliku policy_slices_results/policy_snapshots.npz. "
            "Pominięto porównanie strategii ze składką."
        )

    print("\nWygenerowano wykresy w folderze:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()