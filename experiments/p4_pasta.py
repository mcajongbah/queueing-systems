"""Problem 4: what arrivals actually see (PASTA)."""
import os
import numpy as np

from experiments.common import Context, RESULTS, FIGURES
from experiments import plots
from experiments.p1_mm1 import mm1_spec, BIN_LABELS, MU
from experiments.p3_loss import loss_spec, LAM as LAM3, MU as MU3
from simulator import ClassSpec, SystemSpec, Deterministic
from simulator import theory
from simulator.protocol import write_csv, hist_mean_ci, ci, replication_rows, histogram_rows

TV = theory.total_variation


def _tv_rows(res, pairs, cfg):
    """Per-replication total-variation distances for the given (name, kind_a, kind_b_or_array) pairs."""
    rows = []
    for r in res:
        row = dict(cfg, seed=r["metrics"]["seed"])
        for name, a, b in pairs:
            pa = r["hist"][a] if isinstance(a, str) else a
            pb = r["hist"][b] if isinstance(b, str) else b
            row[name] = TV(pa, pb)
        rows.append(row)
    return rows


def _tv_summary(rows, names, theory_vals, cfg):
    """One summary row per distance: replication mean, CI and the theoretical value."""
    out = []
    for name in names:
        mean, s, half, n = ci([r[name] for r in rows])
        out.append(dict(cfg, distance=name, mean=mean, std=s, ci_half=half, n_reps=n, theory=theory_vals.get(name)))
    return out


def run(ctx: Context):
    print("Problem 4: PASTA")
    rep_rows, occ_rows, tv_summary = [], [], []
    # ------------------------------------------------------------- (a) M/M/1 at rho = 0.8
    lam = 0.8 * MU
    res = ctx.run(mm1_spec(lam))
    th_pn = theory.mm1_pn(lam, MU)
    th_post = np.concatenate([[0.0], th_pn[:15], [th_pn[15:].sum()]])     # post-arrival = pre shifted by one
    cfg = dict(problem="P4a", model="M/M/1", rho=0.8, lam=lam, mu=MU)
    pairs = [("TV_pre_vs_timeavg", "seen_offered", "time_avg"), ("TV_post_vs_timeavg", "seen_post", "time_avg"),
             ("TV_pre_vs_theory", "seen_offered", th_pn), ("TV_timeavg_vs_theory", "time_avg", th_pn),
             ("TV_post_vs_theory", "seen_post", th_pn), ("TV_post_vs_shifted_theory", "seen_post", th_post)]
    rows_a = _tv_rows(res, pairs, cfg)
    write_csv(os.path.join(RESULTS, "p4a_total_variation.csv"), rows_a)
    tv_post_theory = TV(th_post, th_pn)
    tv_summary += _tv_summary(rows_a, [p[0] for p in pairs],
                              {"TV_pre_vs_timeavg": 0.0, "TV_post_vs_timeavg": tv_post_theory,
                               "TV_pre_vs_theory": 0.0, "TV_timeavg_vs_theory": 0.0,
                               "TV_post_vs_theory": tv_post_theory, "TV_post_vs_shifted_theory": 0.0}, cfg)
    for k in ("TV_pre_vs_timeavg", "TV_post_vs_timeavg"):
        m_, s_, h_, _ = ci([r[k] for r in rows_a])
        ctx.note(f"P4a {k}", m_)
        ctx.note(f"P4a {k}_ci", h_)
    ctx.note("P4a TV_post_theory", tv_post_theory)
    dists = []
    for kind, label in (("time_avg", "time average $p_n$"), ("seen_offered", "seen just before arrival"),
                        ("seen_post", "seen just after arrival (incl. itself)")):
        m_, h_ = hist_mean_ci(res, kind)
        dists.append({"label": label, "p": m_, "half": h_})
    plots.grouped_distributions(os.path.join(FIGURES, "fig_p4a_mm1_pasta.png"), BIN_LABELS, dists,
                                "M/M/1, $\\rho = 0.8$: occupancy by sampling rule (mean, 95% CI)", theory=th_pn)

    # ------------------------------------------------------------- (b) D/D/1: one arrival per second, service 0.6 s
    spec_b = SystemSpec([ClassSpec(Deterministic(1.0), Deterministic(0.6), first_arrival=1.0)], servers=1,
                        label="dd1_lam1_x0.6")
    res_b = ctx.run(spec_b, seeds=[ctx.seeds[0]])
    cfg_b = dict(problem="P4b", model="D/D/1", lam=1.0, mu=1 / 0.6, servers=1)
    rep_rows += replication_rows(res_b, cfg_b)
    occ_rows += histogram_rows(res_b, cfg_b)
    h = res_b[0]["hist"]
    mb = res_b[0]["metrics"]
    rows_b = _tv_rows(res_b, [("TV_pre_vs_timeavg", "seen_offered", "time_avg"),
                              ("TV_post_vs_timeavg", "seen_post", "time_avg")], cfg_b)
    tv_summary += _tv_summary(rows_b, ["TV_pre_vs_timeavg", "TV_post_vs_timeavg"],
                              {"TV_pre_vs_timeavg": 0.6, "TV_post_vs_timeavg": 0.4}, cfg_b)
    ctx.note("P4b N", mb["N"])
    ctx.note("P4b lamT", mb["little_N_from_T"])
    ctx.note("P4b P0_timeavg", float(h["time_avg"][0]))
    ctx.note("P4b P0_pre_arrival", float(h["seen_offered"][0]))
    ctx.note("P4b P1_post_arrival", float(h["seen_post"][1]))
    ctx.note("P4b TV_pre", TV(h["seen_offered"], h["time_avg"]))
    plots.grouped_distributions(os.path.join(FIGURES, "fig_p4b_dd1.png"), BIN_LABELS,
                                [{"label": "time average", "p": h["time_avg"]}, {"label": "seen just before arrival", "p": h["seen_offered"]},
                                 {"label": "seen just after arrival", "p": h["seen_post"]}],
                                "D/D/1: one arrival per second, service 0.6 s (single run)", nbins_show=4, figsize=(4.2, 2.8))

    # ------------------------------------------------------------- (c) loss system m = 3: offered vs admitted arrivals
    res3 = ctx.run(loss_spec(3))
    th_all = theory.mmmm_pn(LAM3, MU3, 3)
    th_adm = theory.mmmm_pn_admitted(LAM3, MU3, 3)
    cfg_c = dict(problem="P4c", model="M/M/3/3", lam=LAM3, mu=MU3)
    pairs = [("TV_offered_vs_timeavg", "seen_offered", "time_avg"), ("TV_admitted_vs_timeavg", "seen_admitted", "time_avg"),
             ("TV_offered_vs_theory", "seen_offered", th_all), ("TV_admitted_vs_conditioned_theory", "seen_admitted", th_adm),
             ("TV_timeavg_vs_theory", "time_avg", th_all)]
    rows_c = _tv_rows(res3, pairs, cfg_c)
    write_csv(os.path.join(RESULTS, "p4c_total_variation.csv"), rows_c)
    tv_summary += _tv_summary(rows_c, [p[0] for p in pairs],
                              {"TV_offered_vs_timeavg": 0.0, "TV_admitted_vs_timeavg": TV(th_adm, th_all),
                               "TV_offered_vs_theory": 0.0, "TV_admitted_vs_conditioned_theory": 0.0,
                               "TV_timeavg_vs_theory": 0.0}, cfg_c)
    m_, s_, h_, _ = ci([r["TV_offered_vs_timeavg"] for r in rows_c])
    ctx.note("P4c TV_offered", m_)
    ctx.note("P4c TV_offered_ci", h_)
    m_, s_, h_, _ = ci([r["TV_admitted_vs_timeavg"] for r in rows_c])
    ctx.note("P4c TV_admitted", m_)
    ctx.note("P4c TV_admitted_ci", h_)
    ctx.note("P4c TV_admitted_theory", TV(th_adm, th_all))
    dists = []
    for kind, label in (("time_avg", "time average $p_n$"), ("seen_offered", "seen by all offered arrivals"),
                        ("seen_admitted", "seen by admitted arrivals only")):
        m_, h_ = hist_mean_ci(res3, kind)
        dists.append({"label": label, "p": m_, "half": h_})
    plots.grouped_distributions(os.path.join(FIGURES, "fig_p4c_loss_seen.png"), BIN_LABELS, dists,
                                "M/M/3/3: occupancy seen by offered vs admitted arrivals", theory=th_all,
                                theory_label="Erlang $p_n$ (theory)", nbins_show=4, figsize=(4.4, 2.8), legend_loc="upper left")
    write_csv(os.path.join(RESULTS, "p4_replications.csv"), rep_rows)
    write_csv(os.path.join(RESULTS, "p4_occupancy.csv"), occ_rows)
    write_csv(os.path.join(RESULTS, "p4_total_variation_summary.csv"), tv_summary)
    return {}
