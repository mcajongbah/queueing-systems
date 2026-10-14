"""Problem 5: service-time variability in an M/G/1 queue."""
import os
import numpy as np

from experiments.common import Context, RESULTS, FIGURES
from experiments import plots
from experiments.p1_mm1 import mm1_spec, MU
from simulator import ClassSpec, SystemSpec, Exponential, Deterministic, TwoPoint
from simulator import theory
from simulator.protocol import summarize, replication_rows, histogram_rows, write_csv

LAMS = [200.0, 500.0, 800.0, 950.0]
MODELS = [("exponential", "Exponential", Exponential(1000.0)),
          ("deterministic", "Deterministic 1 ms", Deterministic(1e-3)),
          ("two_point", "Two-point", TwoPoint(0.25e-3, 0.8, 4e-3))]
METRICS = ["svc_mean", "svc_second_moment", "frac_serving", "W", "T", "N_Q", "N",
           "little_N_from_T", "little_NQ_from_W", "prob_sum"]


def spec_for(name, dist, lam):
    if name == "exponential":
        return mm1_spec(lam)                           # identical configuration to Problem 1 -> reused run
    return SystemSpec([ClassSpec(Exponential(lam), dist)], servers=1, label=f"mg1_{name}_lam{lam:g}")


def run(ctx: Context):
    print("Problem 5: M/G/1")
    rep_rows, sum_rows, occ_rows = [], [], []
    series = []
    grid = np.linspace(0.02, 0.97, 200)
    for i, (name, desc, dist) in enumerate(MODELS):
        EX, EX2 = dist.mean, dist.second_moment
        W_mean, W_half = [], []
        for lam in LAMS:
            res = ctx.run(spec_for(name, dist, lam))
            th = theory.mg1(lam, EX, EX2)
            th.update({"svc_mean": EX, "svc_second_moment": EX2, "frac_serving": th["rho"],
                       "little_N_from_T": th["N"], "little_NQ_from_W": th["N_Q"], "prob_sum": 1.0})
            cfg = dict(problem="P5", model=f"M/G/1 {name}", service=dist.describe(), lam=lam, rho=th["rho"],
                       servers=1, capacity="inf", reused_from_P1=(name == "exponential"))
            rep_rows += replication_rows(res, cfg)
            s = summarize(res, th, METRICS, extra=cfg)
            sum_rows += s
            occ_rows += histogram_rows(res, cfg, kinds=("time_avg",))
            S = {r["metric"]: r for r in s}
            W_mean.append(S["W"]["mean"])
            W_half.append(S["W"]["ci_half"])
            ctx.note(f"P5 {name} lam={lam:g} W_ms", S["W"]["mean"] * 1e3)
            ctx.note(f"P5 {name} lam={lam:g} W_relerr_pct", 100 * S["W"]["rel_error"])
        series.append({"label": f"{desc} (simulation)", "x": [l / MU for l in LAMS],
                       "y": np.array(W_mean) * 1e3, "yerr": np.array(W_half) * 1e3, "color": i,
                       "theory_x": grid, "theory_y": 1e3 * (grid * MU) * EX2 / (2 * (1 - grid)),
                       "theory_label": f"{desc} (P-K)"})
    write_csv(os.path.join(RESULTS, "p5_replications.csv"), rep_rows)
    write_csv(os.path.join(RESULTS, "p5_summary.csv"), sum_rows)
    write_csv(os.path.join(RESULTS, "p5_occupancy.csv"), occ_rows)
    plots.metric_vs_x(os.path.join(FIGURES, "fig_p5_W_vs_rho.png"), series,
                      xlabel="utilization $\\rho = \\lambda E[X]$", ylabel="mean queueing delay $W$ (ms)",
                      title="M/G/1: $W$ versus $\\rho$ for three service laws with equal mean", logy=True,
                      legend_loc="upper left")
    return {}
