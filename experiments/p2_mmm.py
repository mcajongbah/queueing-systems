"""Problem 2: M/M/m and M/M/inf."""
import os
import numpy as np

from experiments.common import Context, RESULTS, FIGURES
from experiments import plots
from experiments.p1_mm1 import BIN_LABELS
from simulator import ClassSpec, SystemSpec, Exponential
from simulator import theory
from simulator.protocol import summarize, replication_rows, histogram_rows, write_csv, hist_mean_ci

LAM = 1600.0
MU = 1000.0
SERVERS = [2, 3, 4, None]
METRICS = ["P_Q", "W", "T", "N_Q", "N", "busy_mean", "little_N_from_T", "little_NQ_from_W", "prob_sum"]


def spec_for(m):
    return SystemSpec([ClassSpec(Exponential(LAM), Exponential(MU))], servers=m,
                      label=f"mmm_lam{LAM:g}_mu{MU:g}_m{'inf' if m is None else m}")


def run(ctx: Context):
    print("Problem 2: M/M/m and M/M/inf")
    rep_rows, sum_rows, occ_rows, pn_rows = [], [], [], []
    W_mean, W_half, W_th = [], [], []
    inf_panel = None
    for m in SERVERS:
        res = ctx.run(spec_for(m))
        if m is None:
            th = theory.mminf(LAM, MU)
            th_pn = theory.poisson_pn(LAM / MU)
        else:
            th = theory.erlang_c(LAM, MU, m)
            th_pn = theory.mmm_pn(LAM, MU, m)
        th.update({"little_N_from_T": th["N"], "little_NQ_from_W": th["N_Q"], "prob_sum": 1.0})
        cfg = dict(problem="P2", model="M/M/inf" if m is None else "M/M/m", lam=LAM, mu=MU,
                   servers="inf" if m is None else m, capacity="inf",
                   rho_per_server=(0 if m is None else LAM / (m * MU)))
        rep_rows += replication_rows(res, cfg)
        s = summarize(res, th, METRICS, extra=cfg)
        sum_rows += s
        occ_rows += histogram_rows(res, cfg)
        S = {r["metric"]: r for r in s}
        pmean, phalf = hist_mean_ci(res, "time_avg")
        tv = theory.total_variation(pmean, th_pn)
        for i in range(17):
            pn_rows.append(dict(cfg, bin=BIN_LABELS[i].replace("$\\geq$", ">="), sim_mean=pmean[i],
                                sim_ci_half=phalf[i], theory=th_pn[i], abs_error=pmean[i] - th_pn[i]))
        if m is None:
            inf_panel = {"title": "M/M/$\\infty$, $a = \\lambda/\\mu = 1.6$", "p": pmean, "half": phalf, "theory": th_pn}
        W_mean.append(S["W"]["mean"])
        W_half.append(S["W"]["ci_half"])
        W_th.append(th["W"])
        ctx.note(f"P2 m={m} W_ms", S["W"]["mean"] * 1e3)
        ctx.note(f"P2 m={m} P_Q", S["P_Q"]["mean"])
        ctx.note(f"P2 m={m} TV_pn", tv)
    write_csv(os.path.join(RESULTS, "p2_replications.csv"), rep_rows)
    write_csv(os.path.join(RESULTS, "p2_summary.csv"), sum_rows)
    write_csv(os.path.join(RESULTS, "p2_occupancy.csv"), occ_rows)
    write_csv(os.path.join(RESULTS, "p2_occupancy_summary.csv"), pn_rows)
    plots.metric_vs_x(os.path.join(FIGURES, "fig_p2_W_vs_servers.png"),
                      [{"label": "simulation (mean $\\pm$ 95% CI)", "x": [0, 1, 2, 3], "y": np.array(W_mean) * 1e3,
                        "yerr": np.array(W_half) * 1e3, "theory_x": [0, 1, 2, 3], "theory_y": np.array(W_th) * 1e3,
                        "theory_label": "Erlang C / M/M/$\\infty$", "theory_markers": True, "theory_color": 1}],
                      xlabel="number of servers $m$", ylabel="mean queueing delay $W$ (ms)",
                      title="$W$ versus number of servers ($\\lambda = 1600$/s, $\\mu = 1000$/s)",
                      xticks=[0, 1, 2, 3], xticklabels=["2", "3", "4", "$\\infty$"], ylim_zero=True)
    plots.distribution_panels(os.path.join(FIGURES, "fig_p2_mminf_hist.png"), BIN_LABELS, [inf_panel],
                              suptitle="M/M/$\\infty$ time-average occupancy: simulation (bars, 95% CI) vs Poisson($a$)",
                              ncols=1, width=5.2)
    return {}
