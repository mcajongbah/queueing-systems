"""Problem 6: server vacations and slotted service."""
import os
import numpy as np

from experiments.common import Context, RESULTS, FIGURES
from experiments import plots
from simulator import ClassSpec, SystemSpec, Exponential, Deterministic
from simulator import theory
from simulator.protocol import summarize, replication_rows, histogram_rows, write_csv

LAM = 400.0
X = Deterministic(1e-3)
POLICIES = [("none", None),
            ("det1", Deterministic(1e-3)),
            ("det5", Deterministic(5e-3)),
            ("exp5", Exponential(200.0))]
METRICS = ["W", "T", "frac_serving", "frac_vacation", "frac_empty", "N_Q", "N", "little_N_from_T", "little_NQ_from_W", "prob_sum"]
SLOT = 1e-3


def spec_for(key, vac):
    return SystemSpec([ClassSpec(Exponential(LAM), X)], servers=1, vacation=vac, start_with_vacation=True,
                      label=f"md1_lam{LAM:g}_vac_{key}")


def run(ctx: Context):
    print("Problem 6: vacations")
    rep_rows, sum_rows, occ_rows = [], [], []
    W_mean, W_half, W_th = [], [], []
    rho = LAM * X.mean
    for key, vac in POLICIES:
        res = ctx.run(spec_for(key, vac), keep_records=(key == "det1"))
        if vac is None:
            th = theory.mg1(LAM, X.mean, X.second_moment)
            th.update({"frac_serving": rho, "frac_vacation": 0.0, "frac_empty": 1 - rho})
            residual = 0.0
        else:
            th = theory.mg1_vacation(LAM, X.mean, X.second_moment, vac.mean, vac.second_moment)
            th.update({"frac_empty": None})
            residual = vac.second_moment / (2 * vac.mean)
        th.update({"little_N_from_T": th["N"], "little_NQ_from_W": th["N_Q"], "prob_sum": 1.0})
        cfg = dict(problem="P6", model="M/D/1 with vacations" if vac else "M/D/1", lam=LAM, service=X.describe(),
                   vacation=vac.describe() if vac else "none", residual_vacation_s=residual, servers=1, capacity="inf")
        rep_rows += replication_rows(res, cfg)
        s = summarize(res, th, METRICS, extra=cfg)
        sum_rows += s
        occ_rows += histogram_rows(res, cfg, kinds=("time_avg",))
        S = {r["metric"]: r for r in s}
        W_mean.append(S["W"]["mean"])
        W_half.append(S["W"]["ci_half"])
        W_th.append(th["W"])
        ctx.note(f"P6 {key} W_ms", S["W"]["mean"] * 1e3)
        ctx.note(f"P6 {key} W_ci_ms", S["W"]["ci_half"] * 1e3)
        ctx.note(f"P6 {key} W_th_ms", th["W"] * 1e3)
        ctx.note(f"P6 {key} frac_vacation", S["frac_vacation"]["mean"])
        ctx.note(f"P6 {key} frac_empty", S["frac_empty"]["mean"])
        ctx.note(f"P6 {key} frac_serving", S["frac_serving"]["mean"])
        if key == "det1":
            starts = np.concatenate([r["records"]["start"] for r in res])
            frac = starts / SLOT
            dev = np.abs(frac - np.round(frac))
            ctx.note("P6c n_service_starts", int(len(starts)))
            ctx.note("P6c frac_on_slot_boundary", float(np.mean(dev < 1e-6)))
            ctx.note("P6c max_deviation_ns", float(dev.max() * SLOT * 1e9))
            slot_rows = []
            for r in res:
                st = r["records"]["start"]
                d = np.abs(st / SLOT - np.round(st / SLOT))
                slot_rows.append({"seed": r["metrics"]["seed"], "service_starts": len(st),
                                  "max_abs_deviation_from_slot_boundary_s": float(d.max() * SLOT),
                                  "fraction_on_boundary": float(np.mean(d < 1e-6))})
            write_csv(os.path.join(RESULTS, "p6c_slot_alignment.csv"), slot_rows)
    ctx.note("P6c extra_delay_ms", (W_mean[1] - W_mean[0]) * 1e3)
    ctx.note("P6c extra_delay_theory_ms", X.mean / 2 * 1e3)
    write_csv(os.path.join(RESULTS, "p6_replications.csv"), rep_rows)
    write_csv(os.path.join(RESULTS, "p6_summary.csv"), sum_rows)
    write_csv(os.path.join(RESULTS, "p6_occupancy.csv"), occ_rows)
    plots.metric_vs_x(os.path.join(FIGURES, "fig_p6_W_policies.png"),
                      [{"label": "simulation (mean $\\pm$ 95% CI)", "x": [0, 1, 2, 3], "y": np.array(W_mean) * 1e3,
                        "yerr": np.array(W_half) * 1e3, "theory_x": [0, 1, 2, 3], "theory_y": np.array(W_th) * 1e3,
                        "theory_label": "vacation formula", "theory_markers": True, "theory_color": 1}],
                      xlabel="vacation policy", ylabel="mean queueing delay $W$ (ms)",
                      title="M/D/1 with vacations: $W$ by policy ($\\lambda = 400$/s, $X = 1$ ms)",
                      xticks=[0, 1, 2, 3], xticklabels=["none", "det 1 ms", "det 5 ms", "exp, mean 5 ms"], ylim_zero=True)
    return {}
