"""Problem 3: loss systems, capacity sizing and the two-class truncated chain."""
import os
import numpy as np

from experiments.common import Context, RESULTS, FIGURES
from experiments import plots
from simulator import ClassSpec, SystemSpec, Exponential
from simulator import theory
from simulator.protocol import summarize, replication_rows, histogram_rows, write_csv, t_factor

LAM = 6.0
MU = 2.0
TARGET = 0.01
METRICS_A = ["P_block", "admitted_rate", "offered_rate", "busy_mean", "N", "T", "little_N_from_T", "prob_sum"]
L1, M1 = 2.0, 2.0
L2, M2 = 1.0, 1.0
M_CH = 3


def loss_spec(m, lam=LAM, mu=MU):
    return SystemSpec([ClassSpec(Exponential(lam), Exponential(mu))], servers=m, capacity=m,
                      label=f"mmmm_lam{lam:g}_mu{mu:g}_m{m}")


def two_class_spec():
    return SystemSpec([ClassSpec(Exponential(L1), Exponential(M1), name="class1"),
                       ClassSpec(Exponential(L2), Exponential(M2), name="class2")],
                      servers=M_CH, capacity=M_CH, label="loss_two_class_m3")


def loss_theory(m):
    th = theory.mmmm(LAM, MU, m)
    th.update({"offered_rate": LAM, "little_N_from_T": th["N"], "prob_sum": 1.0})
    return th


def run(ctx: Context):
    print("Problem 3: loss systems")
    rep_rows, sum_rows, occ_rows = [], [], []
    # ---------------------------------------------------------------- (a)
    res3 = ctx.run(loss_spec(3))
    th3 = loss_theory(3)
    cfg = dict(problem="P3a", model="M/M/m/m", lam=LAM, mu=MU, servers=3, capacity=3)
    rep_rows += replication_rows(res3, cfg)
    s3 = summarize(res3, th3, METRICS_A, extra=cfg)
    sum_rows += s3
    occ_rows += histogram_rows(res3, cfg)
    S3 = {r["metric"]: r for r in s3}
    for k in ("P_block", "admitted_rate", "busy_mean", "T", "little_N_from_T"):
        ctx.note(f"P3a {k}", S3[k]["mean"])
        ctx.note(f"P3a {k}_ci", S3[k]["ci_half"])
    ctx.note("P3a offered_rate_times_T (wrong Little rate)", S3["offered_rate"]["mean"] * S3["T"]["mean"])

    # ---------------------------------------------------------------- (b)
    ms = list(range(1, 13))
    B = [theory.erlang_b(LAM / MU, m) for m in ms]
    m_star = next(m for m, b in zip(ms, B) if b <= TARGET)
    ctx.note("P3b m_star", m_star)
    sim_points = []
    for m in (m_star - 1, m_star):
        res = ctx.run(loss_spec(m))
        th = loss_theory(m)
        cfg = dict(problem="P3b", model="M/M/m/m", lam=LAM, mu=MU, servers=m, capacity=m)
        rep_rows += replication_rows(res, cfg)
        s = summarize(res, th, METRICS_A, extra=cfg)
        sum_rows += s
        occ_rows += histogram_rows(res, cfg)
        pb = {r["metric"]: r for r in s}["P_block"]
        sim_points.append((m, pb["mean"], pb["ci_half"]))
        ctx.note(f"P3b m={m} P_block", pb["mean"])
        ctx.note(f"P3b m={m} P_block_ci", pb["ci_half"])
        ctx.note(f"P3b m={m} CI_below_target", bool(pb["ci_high"] < TARGET))
    write_csv(os.path.join(RESULTS, "p3b_erlangB_curve.csv"),
              [{"m": m, "erlang_B": b, "meets_1pct": b <= TARGET} for m, b in zip(ms, B)])
    plots.sizing_curve(os.path.join(FIGURES, "fig_p3_erlangB_sizing.png"), ms, B, TARGET, sim_points,
                       title="Capacity sizing: Erlang B for $a = 3$ Erlangs with simulated points")

    # ---------------------------------------------------------------- (c)
    res2 = ctx.run(two_class_spec())
    th2 = theory.two_class_loss(L1, M1, L2, M2, M_CH)
    for r in res2:                       # per-replication Little checks N_i ~ lambda_i,eff / mu_i
        m = r["metrics"]
        m["N_c1_from_little"] = m["admitted_rate_c1"] / M1
        m["N_c2_from_little"] = m["admitted_rate_c2"] / M2
    th2.update({"prob_sum": 1.0})
    cfg = dict(problem="P3c", model="two-class M/M/3/3", lam=L1 + L2, mu="2,1", servers=3, capacity=3)
    rep_rows += replication_rows(res2, cfg)
    s2 = summarize(res2, th2, ["P_block_c1", "P_block_c2", "admitted_rate_c1", "admitted_rate_c2",
                               "N_c1", "N_c2", "N_c1_from_little", "N_c2_from_little", "N", "prob_sum"], extra=cfg)
    sum_rows += s2
    S2 = {r["metric"]: r for r in s2}
    states = sorted(th2["pi"].keys())
    J = np.array([[r["joint"].get(s, 0.0) for s in states] for r in res2])
    jmean = J.mean(axis=0)
    jhalf = (t_factor(J.shape[0]) * J.std(axis=0, ddof=1) / np.sqrt(J.shape[0])
             if J.shape[0] > 1 else np.zeros_like(jmean))
    joint_rows = []
    for s, mu_, h in zip(states, jmean, jhalf):
        joint_rows.append({"problem": "P3c", "n1": s[0], "n2": s[1], "sim_mean": mu_, "sim_ci_half": h,
                           "theory": th2["pi"][s], "abs_error": mu_ - th2["pi"][s],
                           "aggregate_departure_rate": s[0] * M1 + s[1] * M2})
    write_csv(os.path.join(RESULTS, "p3c_joint_states.csv"), joint_rows)
    for i in (1, 2):
        ctx.note(f"P3c class{i} P_block", S2[f"P_block_c{i}"]["mean"])
        ctx.note(f"P3c class{i} N", S2[f"N_c{i}"]["mean"])
        ctx.note(f"P3c class{i} N_from_little", S2[f"N_c{i}_from_little"]["mean"])
    sim_mat = np.full((M_CH + 1, M_CH + 1), np.nan)
    th_mat = np.full((M_CH + 1, M_CH + 1), np.nan)
    for s, mu_ in zip(states, jmean):
        sim_mat[s[0], s[1]] = mu_
        th_mat[s[0], s[1]] = th2["pi"][s]
    plots.heatmap_pair(os.path.join(FIGURES, "fig_p3_two_class_heatmap.png"), sim_mat, th_mat,
                       title="Two-class loss system: time-average state probabilities")
    # (d) states with equal total occupancy but different aggregate departure rates
    ctx.note("P3d rates", {f"({s[0]},{s[1]})": s[0] * M1 + s[1] * M2 for s in states if s[0] + s[1] == M_CH})

    write_csv(os.path.join(RESULTS, "p3_replications.csv"), rep_rows)
    write_csv(os.path.join(RESULTS, "p3_summary.csv"), sum_rows)
    write_csv(os.path.join(RESULTS, "p3_occupancy.csv"), occ_rows)
    return {"res3": res3}
