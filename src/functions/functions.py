import numpy as np


def S_exp(H, v, a_1):
    """
    Funkcja prawdopodobieństwa wycieku S_1(H,v).
    """
    return v * np.exp(-a_1 * H)


def S_rational(H, v, a_2, b):
    """
    Funkcja prawdopodobieństwa wycieku S_2(H,v).
    """
    return v / (1.0 + a_2 * H) ** b


def premium(S_value, v, pi_0, r):
    """
    Składka chwilowa pi(H;d,v,r).
    """
    return pi_0 * (1.0 - r * (1.0 - S_value / v))


def investment_cost(z, zeta, gamma):
    """
    Koszt inwestycji c(z).
    """
    return zeta * z + 0.5 * gamma * z ** 2


def terminal_utility(H):
    """
    Końcowa użyteczność z poziomu zabezpieczeń.
    """
    return np.sqrt(H)