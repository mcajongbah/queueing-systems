"""Common experimental protocol: replications, confidence intervals, CSV output.

Replications start empty and use independent seeds.  With total offered rate
lambda, warm-up = WARM_MULT/lambda seconds and observation window
H = OBS_MULT/lambda seconds (defaults 10,000 and 100,000).  The system state is
kept at the end of warm-up; only the counters are reset (they only ever count
inside the window).  Confidence intervals use the 10-replication factor 2.262
from the assignment (Student t, 9 degrees of freedom, 95%).
"""
import csv
import math
import multiprocessing as mp
import os
from math import sqrt

import numpy as np

from .system import QueueingSystem

WARM_MULT = 10_000.0
OBS_MULT = 100_000.0
T_FACTOR_10 = 2.262
METRIC_ORDER = [
    "N", "N_Q", "W", "T", "busy_mean", "frac_serving", "frac_vacation", "frac_empty",
    "P_Q", "P_block", "admitted_rate", "offered_rate", "little_N_from_T", "little_NQ_from_W",
    "svc_mean", "svc_second_moment", "prob_sum",
]


def t_factor(n):
    if n == 10:
        return T_FACTOR_10
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except Exception:            # pragma: no cover
        return 1.96


def _run_one(args):
    spec, seed, t_warm, t_obs, keep_records = args
    sim = QueueingSystem(spec, seed, t_warm, t_obs, keep_records=keep_records)
    return sim.run()


def run_replications(spec, seeds, warm_mult=WARM_MULT, obs_mult=OBS_MULT,
                     keep_records=False, workers=None):
    """Run one configuration for every seed (in parallel) and return the list of results."""
    lam = spec.total_rate
    t_warm = warm_mult / lam
    t_obs = obs_mult / lam
    jobs = [(spec, int(s), t_warm, t_obs, keep_records) for s in seeds]
    if workers is None:
        workers = int(os.environ.get("SIMULATOR_WORKERS", max(1, min(len(jobs), os.cpu_count() or 1))))
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_run_one, jobs)
    else:
        results = [_run_one(j) for j in jobs]
    for r in results:
        r["metrics"]["label"] = spec.label
        r["metrics"]["spec"] = spec.describe()
    return results


def ci(values):
    """Mean, sample std, and 95% half-width across replication-level estimates."""
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))],
                   dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(v.mean())
    s = float(v.std(ddof=1)) if n > 1 else 0.0
    half = t_factor(n) * s / sqrt(n) if n > 1 else float("nan")
    return mean, s, half, n


def summarize(results, theory, metrics, label=None, extra=None):
    """Build summary rows (one per metric) with mean, CI, theory and errors."""
    rows = []
    for key in metrics:
        vals = [r["metrics"].get(key) for r in results]
        mean, s, half, n = ci(vals)
        th = theory.get(key) if theory else None
        row = {"label": label if label is not None else results[0]["metrics"].get("label", ""),
               "metric": key, "mean": mean, "std": s, "ci_half": half, "ci_low": mean - half,
               "ci_high": mean + half, "n_reps": n, "theory": th}
        if th is not None and not (isinstance(th, float) and math.isnan(th)):
            row["abs_error"] = mean - th
            row["rel_error"] = (mean - th) / th if th != 0 else float("nan")
            row["theory_in_ci"] = (abs(mean - th) <= half) if n > 1 else None
        else:
            row["abs_error"] = row["rel_error"] = float("nan")
            row["theory_in_ci"] = None
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def replication_rows(results, config_fields):
    """Flatten per-replication metrics into CSV rows, prefixed with configuration fields."""
    rows = []
    for r in results:
        row = dict(config_fields)
        row.update(r["metrics"])
        rows.append(row)
    return rows


def histogram_rows(results, config_fields, kinds=("time_avg", "seen_offered", "seen_admitted", "seen_post")):
    rows = []
    for r in results:
        for kind in kinds:
            h = r["hist"][kind]
            row = dict(config_fields)
            row.update({"seed": r["metrics"]["seed"], "kind": kind})
            for i, v in enumerate(h):
                row[f"bin{i}" if i < 16 else "bin16plus"] = float(v)
            rows.append(row)
    return rows


def hist_mean_ci(results, kind):
    """Replication-averaged histogram with CI half-widths per bin."""
    H = np.array([r["hist"][kind] for r in results])
    mean = H.mean(axis=0)
    n = H.shape[0]
    half = t_factor(n) * H.std(axis=0, ddof=1) / sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, half
