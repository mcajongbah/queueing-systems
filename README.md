# ECE 533 - Homework 2: Implementing Queueing Systems

One event-driven queueing simulator (`simulator/`) reused across the six problems,
plus the experiment scripts, results, figures and the LaTeX report.

## Reproduce everything

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pytest
python -m pytest tests -q                # event-handling and conservation tests (~1 s)
python run_all.py                        # all experiments -> results/ and figures/  (~1 min on 10 cores)
```

`python run_all.py --quick` runs a 3-seed, 10x shorter smoke test in ~15 s.
Options: `--problems 1 4`, `--reps N`, `--seed-base S`, `--warm-mult`, `--obs-mult`,
`--workers`, `--no-cache` (simulation results are cached in `results/cache/` so that later
problems reuse earlier runs, e.g. Problem 5 reuses the exponential-service runs of Problem 1).

The report (`report/report.pdf`, source `report/report.tex` with its tables in `report/tables/`)
is a static document produced from the results of the full run; the scripts do not modify it.

Any single configuration can be simulated from the command line, e.g.

```bash
python simulate.py --rate 1600 --service exp:1000 --servers 3
python simulate.py --rate 6 --service exp:2 --servers 3 --capacity 3 --extra-class 1,exp:1
python simulate.py --rate 400 --service det:0.001 --vacation exp:200 --reps 10 --csv out.csv
```

Distribution specs are `exp:RATE`, `det:VALUE` and `two:A,P,B` (seconds, rates per second).

## Layout

| path                                          | content |
|-----------------------------------------------|---|
| `simulator/engine.py`                         | future-event list (heap keyed by time, priority, sequence); tie rule departure < vacation end < arrival |
| `simulator/system.py`                         | the simulator: arrival / departure / vacation-end handlers, FCFS queue, admission control, m or infinitely many servers |
| `simulator/distributions.py`                  | exponential, deterministic, two-point generators with exact moments; one independent random stream per process |
| `simulator/stats.py`                          | window-restricted counters: time-weighted occupancy, arrival-seen states, per-customer records |
| `simulator/protocol.py`                       | replications (parallel), confidence intervals (2.262 s/sqrt(10)), CSV writers |
| `simulator/theory.py`                         | analytical references (M/M/1, Erlang C, M/M/inf, Erlang B, product form, P-K, vacations) - never used by the simulator |
| `experiments/p1_mm1.py` ... `p6_vacations.py` | one script per problem |
| `experiments/plots.py`                        | figure helpers (matplotlib) |
| `tests/test_engine.py`                        | tie handling, no service restart, conservation, loss, multi-server, vacations, moments |
| `results/*_replications.csv`                  | one row per configuration and replication (seed, warm-up, window, counts, metrics) |
| `results/*_summary.csv`                       | replication means, 95% CI, theory, absolute and relative errors |
| `results/*_occupancy*.csv`                    | time-average and arrival-seen occupancy distributions (bins 0..15 and >=16) |
| `results/key_numbers.json`                    | the numbers quoted in the report |
| `figures/*.png`                               | all figures |
| `report/report.tex`, `report/tables/*.tex`    | the report and its auto-generated tables |

## Conventions

* Seconds internally; tables report ms where stated.
* Replication k of a configuration uses seed `seed_base + k` (1..10 by default); Problem 1(d) uses seeds 11..20.
* Each run starts empty; warm-up 10,000/lambda s, window 100,000/lambda s, where lambda is the total offered rate.
  State is kept at the end of warm-up; only measurement counters are restricted to the window.
* Time averages are clipped to the window. W and T use admitted customers that arrived inside the window,
  followed until they depart (the system keeps running normally after the window; the time-average window is not extended).
* Blocking and arrival-seen states use arrivals in the window; the state seen by an arrival is recorded before its
  admission decision, and blocked arrivals count in the offered sample.
* Ties: departures are processed before vacation ends, which are processed before arrivals at the same instant.
