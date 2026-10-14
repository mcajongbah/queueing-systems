#!/usr/bin/env python3
"""Run one arbitrary configuration of the simulator from the command line.

Examples
    python simulate.py --rate 800 --service exp:1000
    python simulate.py --rate 1600 --service exp:1000 --servers 3
    python simulate.py --rate 6 --service exp:2 --servers 3 --capacity 3
    python simulate.py --rate 2 --service exp:2 --extra-class 1,exp:1 --servers 3 --capacity 3
    python simulate.py --rate 400 --service det:0.001 --vacation det:0.001 --reps 10 --csv out.csv
    python simulate.py --interarrival det:1 --service det:0.6 --reps 1

Distribution specs: exp:RATE (per second), det:VALUE (seconds), two:A,P,B (A w.p. P, B w.p. 1-P, seconds).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import ClassSpec, SystemSpec, Exponential
from simulator.distributions import parse_distribution
from simulator.protocol import run_replications, summarize, write_csv, replication_rows, METRIC_ORDER, WARM_MULT, OBS_MULT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, help="Poisson arrival rate of class 1 (per second)")
    ap.add_argument("--interarrival", help="interarrival distribution of class 1 instead of --rate (e.g. det:1.0)")
    ap.add_argument("--service", required=True, help="service-time distribution of class 1")
    ap.add_argument("--extra-class", action="append", default=[], metavar="RATE,SERVICE",
                    help="additional Poisson class, e.g. 1,exp:1 (repeatable)")
    ap.add_argument("--servers", default="1", help="number of servers or 'inf'")
    ap.add_argument("--capacity", default="inf", help="max customers in system (loss beyond) or 'inf'")
    ap.add_argument("--vacation", help="vacation-length distribution (enables exhaustive repeated vacations)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=1)
    ap.add_argument("--warm-mult", type=float, default=WARM_MULT)
    ap.add_argument("--obs-mult", type=float, default=OBS_MULT)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--csv", help="write per-replication results to this CSV file")
    args = ap.parse_args(argv)
    if args.interarrival:
        inter = parse_distribution(args.interarrival)
        first = inter.mean
    elif args.rate:
        inter = Exponential(args.rate)
        first = None
    else:
        ap.error("give --rate or --interarrival")
    classes = [ClassSpec(inter, parse_distribution(args.service), name="class1", first_arrival=first)]
    for i, spec in enumerate(args.extra_class, start=2):
        rate, svc = spec.split(",", 1)
        classes.append(ClassSpec(Exponential(float(rate)), parse_distribution(svc), name=f"class{i}"))
    servers = None if args.servers.lower() == "inf" else int(args.servers)
    capacity = None if args.capacity.lower() == "inf" else int(args.capacity)
    vac = parse_distribution(args.vacation) if args.vacation else None
    spec = SystemSpec(classes, servers=servers, capacity=capacity, vacation=vac, label="cli")
    seeds = list(range(args.seed_base, args.seed_base + args.reps))
    print("configuration:", spec.describe())
    print(f"warm-up {args.warm_mult / spec.total_rate:.4g} s, window {args.obs_mult / spec.total_rate:.4g} s, seeds {seeds}")
    results = run_replications(spec, seeds, args.warm_mult, args.obs_mult, workers=args.workers)
    keys = [k for k in METRIC_ORDER if k in results[0]["metrics"]]
    keys += [k for k in results[0]["metrics"] if k.endswith(("_c1", "_c2", "_c3")) and k not in keys]
    rows = summarize(results, {}, keys)
    print(f"{'metric':<22}{'mean':>14}{'95% CI half':>14}{'std':>14}")
    for r in rows:
        print(f"{r['metric']:<22}{r['mean']:>14.6g}{r['ci_half']:>14.3g}{r['std']:>14.3g}")
    cons = all(r["metrics"]["conservation_ok"] for r in results)
    print("conservation identity (admitted - departed == occupancy) holds in every replication:", cons)
    if args.csv:
        write_csv(args.csv, replication_rows(results, {"spec": spec.describe()}))
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
