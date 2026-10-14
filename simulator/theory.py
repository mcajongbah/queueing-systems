"""Analytical reference formulas (page 5 of the assignment).

These are computed independently of the simulator and used only as
validation benchmarks.  All rates are per second, all times in seconds.
"""
from math import factorial, exp
import numpy as np

from .stats import NBINS, OVERFLOW


def _bins(pn_fn, nmax=10_000):
    """Collapse a distribution p_n (callable) into bins 0..15 and >=16."""
    p = np.zeros(NBINS)
    for n in range(OVERFLOW):
        p[n] = pn_fn(n)
    p[OVERFLOW] = max(0.0, 1.0 - p[:OVERFLOW].sum())
    return p


# ----------------------------------------------------------------- M/M/1
def mm1(lam, mu):
    if lam >= mu:
        raise ValueError("M/M/1 stationary formulas require lambda < mu")
    rho = lam / mu
    T = 1.0 / (mu - lam)
    return {"rho": rho, "N": rho / (1 - rho), "N_Q": rho ** 2 / (1 - rho),
            "T": T, "W": T - 1.0 / mu, "utilization": rho, "P_Q": rho,
            "frac_empty": 1 - rho}


def mm1_pn(lam, mu):
    rho = lam / mu
    return _bins(lambda n: (1 - rho) * rho ** n)


# --------------------------------------------------------- M/M/m, Erlang C
def erlang_c(lam, mu, m):
    a = lam / mu
    rho = lam / (m * mu)
    if rho >= 1:
        raise ValueError("M/M/m requires lambda < m*mu")
    p0 = 1.0 / (sum(a ** n / factorial(n) for n in range(m)) + a ** m / (factorial(m) * (1 - rho)))
    PQ = p0 * a ** m / (factorial(m) * (1 - rho))
    W = PQ / (m * mu - lam)
    T = W + 1.0 / mu
    return {"rho": rho, "p0": p0, "P_Q": PQ, "W": W, "T": T, "N_Q": lam * W, "N": lam * T,
            "busy_mean": a}


def mmm_pn(lam, mu, m):
    a = lam / mu
    rho = lam / (m * mu)
    c = erlang_c(lam, mu, m)
    p0 = c["p0"]

    def pn(n):
        if n < m:
            return p0 * a ** n / factorial(n)
        return p0 * a ** m / factorial(m) * rho ** (n - m)
    return _bins(pn)


# ----------------------------------------------------------------- M/M/inf
def mminf(lam, mu):
    a = lam / mu
    return {"N": a, "N_Q": 0.0, "W": 0.0, "T": 1.0 / mu, "P_Q": 0.0, "busy_mean": a}


def poisson_pn(a):
    return _bins(lambda n: exp(-a) * a ** n / factorial(n))


# ------------------------------------------------------- M/M/m/m, Erlang B
def erlang_b(a, m):
    """Blocking probability by the stable recursion B_0 = 1, B_m = a B_{m-1} / (m + a B_{m-1})."""
    B = 1.0
    for k in range(1, m + 1):
        B = a * B / (k + a * B)
    return B


def mmmm(lam, mu, m):
    a = lam / mu
    B = erlang_b(a, m)
    lam_eff = lam * (1 - B)
    return {"a": a, "P_block": B, "admitted_rate": lam_eff, "N": lam_eff / mu, "T": 1.0 / mu,
            "busy_mean": lam_eff / mu}


def mmmm_pn(lam, mu, m):
    a = lam / mu
    G = sum(a ** n / factorial(n) for n in range(m + 1))
    return _bins(lambda n: (a ** n / factorial(n)) / G if n <= m else 0.0)


def mmmm_pn_admitted(lam, mu, m):
    """Distribution seen by ADMITTED arrivals: the Erlang distribution conditioned on n < m."""
    p = mmmm_pn(lam, mu, m)
    q = p.copy()
    q[m:] = 0.0
    return q / q.sum()


# ------------------------------------------------- two-class loss system
def two_class_loss(lam1, mu1, lam2, mu2, m):
    a1, a2 = lam1 / mu1, lam2 / mu2
    states = [(n1, n2) for n1 in range(m + 1) for n2 in range(m + 1 - n1)]
    weights = {s: a1 ** s[0] / factorial(s[0]) * a2 ** s[1] / factorial(s[1]) for s in states}
    G = sum(weights.values())
    pi = {s: w / G for s, w in weights.items()}
    PB = sum(p for s, p in pi.items() if s[0] + s[1] == m)
    N1 = sum(s[0] * p for s, p in pi.items())
    N2 = sum(s[1] * p for s, p in pi.items())
    return {"pi": pi, "P_block_c1": PB, "P_block_c2": PB,
            "admitted_rate_c1": lam1 * (1 - PB), "admitted_rate_c2": lam2 * (1 - PB),
            "N_c1": N1, "N_c2": N2, "N": N1 + N2,
            "N_c1_from_little": lam1 * (1 - PB) / mu1, "N_c2_from_little": lam2 * (1 - PB) / mu2}


# ------------------------------------------------------ M/G/1 and vacations
def mg1(lam, EX, EX2):
    rho = lam * EX
    if rho >= 1:
        raise ValueError("M/G/1 requires rho < 1")
    W = lam * EX2 / (2 * (1 - rho))
    T = W + EX
    return {"rho": rho, "W": W, "T": T, "N_Q": lam * W, "N": lam * T, "utilization": rho,
            "frac_empty": 1 - rho}


def mg1_vacation(lam, EX, EX2, EV, EV2):
    base = mg1(lam, EX, EX2)
    W = base["W"] + EV2 / (2 * EV)
    T = W + EX
    return {"rho": base["rho"], "W": W, "T": T, "N_Q": lam * W, "N": lam * T,
            "frac_serving": base["rho"], "frac_vacation": 1 - base["rho"]}


def total_variation(p, q):
    """One half of the sum of absolute differences across bins."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())
