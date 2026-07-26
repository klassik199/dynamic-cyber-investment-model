import numpy as np


def S_exp(H, v, a_1):
    """Exponential leakage probability S_1(H, v)."""
    return v * np.exp(-a_1 * H)


def S_rational(H, v, a_2, b):
    """Rational leakage probability S_2(H, v)."""
    return v / (1.0 + a_2 * H) ** b


def premium(S_value, v, pi_0, r):
    """Insurance premium adjusted for the relative leakage risk."""
    return pi_0 * (1.0 - r * (1.0 - S_value / v))


def investment_cost(z, zeta, gamma):
    """Running investment cost c(z)."""
    return zeta * z + 0.5 * gamma * z**2


def terminal_utility(H):
    """Terminal utility from the final security level."""
    return np.sqrt(H)
