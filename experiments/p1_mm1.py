"""Problem 1: build and validate an M/M/1 simulator."""
import os
import numpy as np

from experiments.common import Context, RESULTS, FIGURES
from experiments import plots
from simulator import ClassSpec, SystemSpec, Exponential
from simulator import theory
from simulator.protocol import summarize, replication_rows, histogram_rows, write_csv, hist_mean_ci

MU = 1000.0
RHOS = [0.20, 0.50, 0.80, 0.95]
METRICS = ["N", "N_Q", "W", "T", "frac_serving", "P_Q", "frac_empty", "prob_sum",
           "little_N_from_T", "little_NQ_from_W", "admitted_rate"]
BIN_LABELS = [str(i) for i in range(16)] + ["$\\geq$16"]


def mm1_spec(lam, mu=MU, label=None):
    return SystemSpec([ClassSpec(Exponential(lam), Exponential(mu), name="c1")], servers=1,
                      label=label or f"mm1_lam{lam:g}_mu{mu:g}")


def mm1_theory(lam, mu=MU):
    th = theory.mm1(lam, mu)
    th.update({"admitted_rate": lam, "prob_sum": 1.0, "frac_serving": th["rho"],
               "little_N_from_T": th["N"], "little_NQ_from_W": th["N_Q"]})
    return th


def run(ctx: Context):
    print("Problem 1: M/M/1")
    rep_rows, sum_rows, occ_rows, pn_rows = [], [], [], []
    T_mean, T_half = [], []
    pn_panels = []
    for rho in RHOS:
        lam = rho * MU
        res = ctx.run(mm1_spec(lam))
        th = mm1_theory(lam)
        cfg = dict(problem="P1", model="M/M/1", rho=rho, lam=lam, mu=MU, servers=1, capacity="inf")
        rep_rows += replication_rows(res, cfg)
        s = summarize(res, th, METRICS, extra=cfg)
        sum_rows += s
        occ_rows += histogram_rows(res, cfg)
        S = {r["metric"]: r for r in s}
        pmean, phalf = hist_mean_ci(res, "time_avg")
        th_pn = theory.mm1_pn(lam, MU)
        for i in range(17):
            pn_rows.append(dict(cfg, bin=BIN_LABELS[i].replace("$\\geq$", ">="), sim_mean=pmean[i],
                                sim_ci_half=phalf[i], theory=th_pn[i], abs_error=pmean[i] - th_pn[i]))
        pn_panels.append({"title": f"$\\rho$ = {rho:.2f}", "p": pmean, "half": phalf, "theory": th_pn})
        T_mean.append(S["T"]["mean"])
        T_half.append(S["T"]["ci_half"])
        ctx.note(f"P1 rho={rho} T_ms", S["T"]["mean"] * 1e3)
        ctx.note(f"P1 rho={rho} T_ci_ms", S["T"]["ci_half"] * 1e3)
        ctx.note(f"P1 rho={rho} T_relerr_pct", 100 * S["T"]["rel_error"])
        ctx.note(f"P1 rho={rho} conservation_ok", all(r["metrics"]["conservation_ok"] for r in res))

    # ---- (d) rho = 0.95 with doubled warm-up and window, fresh seeds
    lam = 0.95 * MU
    spec_d = mm1_spec(lam, label="mm1_lam950_mu1000_double")
    res_d = ctx.run(spec_d, seeds=ctx.seeds_alt, warm_mult=2 * ctx.warm_mult, obs_mult=2 * ctx.obs_mult)
    th = mm1_theory(lam)
    cfg = dict(problem="P1d", model="M/M/1", rho=0.95, lam=lam, mu=MU, servers=1, capacity="inf")
    rep_rows += replication_rows(res_d, cfg)
    s_d = summarize(res_d, th, METRICS, extra=cfg)
    sum_rows += s_d
    occ_rows += histogram_rows(res_d, cfg)
    S_d = {r["metric"]: r for r in s_d}
    s_o = {r["metric"]: r for r in summarize(ctx.run(mm1_spec(lam)), th, ["T", "N", "W"])}
    ctx.note("P1d original T_ms", s_o["T"]["mean"] * 1e3)
    ctx.note("P1d original T_ci_ms", s_o["T"]["ci_half"] * 1e3)
    ctx.note("P1d doubled T_ms", S_d["T"]["mean"] * 1e3)
    ctx.note("P1d doubled T_ci_ms", S_d["T"]["ci_half"] * 1e3)
    ctx.note("P1d original relerr_pct", 100 * s_o["T"]["rel_error"])
    ctx.note("P1d doubled relerr_pct", 100 * S_d["T"]["rel_error"])

    # ---- files
    write_csv(os.path.join(RESULTS, "p1_replications.csv"), rep_rows)
    write_csv(os.path.join(RESULTS, "p1_summary.csv"), sum_rows)
    write_csv(os.path.join(RESULTS, "p1_occupancy.csv"), occ_rows)
    write_csv(os.path.join(RESULTS, "p1_occupancy_summary.csv"), pn_rows)

    # ---- figures
    grid = np.linspace(0.02, 0.97, 200)
    plots.metric_vs_x(os.path.join(FIGURES, "fig_p1_T_vs_rho.png"),
                      [{"label": "simulation (mean $\\pm$ 95% CI)", "x": RHOS, "y": np.array(T_mean) * 1e3,
                        "yerr": np.array(T_half) * 1e3,
                        "theory_x": grid, "theory_y": 1e3 / (MU - grid * MU),
                        "theory_label": "theory $1/(\\mu-\\lambda)$", "theory_color": 1}],
                      xlabel="utilization $\\rho = \\lambda/\\mu$", ylabel="mean time in system $T$ (ms)",
                      title="M/M/1: simulated $T$ versus $\\rho$")
    plots.distribution_panels(os.path.join(FIGURES, "fig_p1_pn.png"), BIN_LABELS, pn_panels,
                              suptitle="M/M/1 time-average occupancy $p_n$: simulation (bars, 95% CI) and theory (markers)")
    return {"T_mean": T_mean, "T_half": T_half}
