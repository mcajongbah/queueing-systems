#!/usr/bin/env python3
"""Regenerate every experiment, results file and figure of Homework 2.

    python run_all.py                  # full protocol: 10 seeds, warm-up 1e4/lambda, window 1e5/lambda
    python run_all.py --problems 1 4   # a subset (dependencies are re-simulated or taken from the cache)
    python run_all.py --quick          # smoke test: 3 seeds and 10x shorter runs
    python run_all.py --no-cache       # ignore cached simulation results

Simulation output goes to results/ (CSV files and key_numbers.json) and figures/ (PNG).
The report in report/ is a static document and is not touched by this script.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.common import Context, RESULTS      # noqa: E402
from simulator.protocol import WARM_MULT, OBS_MULT   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--reps", type=int, default=10, help="replications (seeds) per configuration")
    ap.add_argument("--seed-base", type=int, default=1, help="first seed; replication k uses seed base+k")
    ap.add_argument("--warm-mult", type=float, default=WARM_MULT, help="warm-up = warm-mult / lambda seconds")
    ap.add_argument("--obs-mult", type=float, default=OBS_MULT, help="window = obs-mult / lambda seconds")
    ap.add_argument("--workers", type=int, default=None, help="parallel worker processes (default: all cores)")
    ap.add_argument("--no-cache", action="store_true", help="re-simulate even if cached results exist")
    ap.add_argument("--quick", action="store_true", help="3 replications and 10x shorter runs (smoke test)")
    args = ap.parse_args(argv)
    if args.quick:
        args.reps = 3
        args.warm_mult /= 10
        args.obs_mult /= 10
    ctx = Context(reps=args.reps, seed_base=args.seed_base, warm_mult=args.warm_mult, obs_mult=args.obs_mult,
                  workers=args.workers, use_cache=not args.no_cache)
    from experiments import p1_mm1, p2_mmm, p3_loss, p4_pasta, p5_mg1, p6_vacations
    modules = {1: p1_mm1, 2: p2_mmm, 3: p3_loss, 4: p4_pasta, 5: p5_mg1, 6: p6_vacations}
    t0 = time.time()
    for p in args.problems:
        tp = time.time()
        modules[p].run(ctx)
        print(f"  problem {p} done in {time.time() - tp:.1f} s", flush=True)
    ctx.save_numbers()
    print(f"all done in {time.time() - t0:.1f} s; results in {RESULTS}")


if __name__ == "__main__":
    main()
