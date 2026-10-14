"""Shared experiment context: seeds, run lengths, caching, output directories."""
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator.protocol import run_replications, WARM_MULT, OBS_MULT  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
CACHE = os.path.join(RESULTS, "cache")


class Context:
    """Holds the protocol parameters and caches simulation results by configuration.

    The cache lets later problems reuse earlier runs (Problem 5 reuses the
    exponential-service runs of Problem 1; Problem 4 reuses Problem 1 and 3) without
    re-simulating, exactly as the assignment asks.
    """

    def __init__(self, reps=10, seed_base=1, warm_mult=WARM_MULT, obs_mult=OBS_MULT,
                 workers=None, use_cache=True):
        self.reps = reps
        self.seed_base = seed_base
        self.seeds = list(range(seed_base, seed_base + reps))
        self.seeds_alt = list(range(seed_base + reps, seed_base + 2 * reps))
        self.warm_mult = warm_mult
        self.obs_mult = obs_mult
        self.workers = workers
        self.use_cache = use_cache
        self._fresh = set()        # cache entries written during this run (always reusable)
        self.numbers = {}          # key numbers collected across problems (results/key_numbers.json)
        for d in (RESULTS, FIGURES, CACHE):
            os.makedirs(d, exist_ok=True)

    def run(self, spec, seeds=None, warm_mult=None, obs_mult=None, keep_records=False):
        seeds = self.seeds if seeds is None else seeds
        warm_mult = self.warm_mult if warm_mult is None else warm_mult
        obs_mult = self.obs_mult if obs_mult is None else obs_mult
        key = f"{spec.label}__s{seeds[0]}-{seeds[-1]}__w{warm_mult:g}__o{obs_mult:g}__r{int(keep_records)}"
        path = os.path.join(CACHE, key + ".pkl")
        if (self.use_cache or key in self._fresh) and os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        print(f"  simulating {spec.label}: {len(seeds)} replications, "
              f"warm-up {warm_mult / spec.total_rate:.3g} s, window {obs_mult / spec.total_rate:.3g} s",
              flush=True)
        results = run_replications(spec, seeds, warm_mult, obs_mult, keep_records=keep_records,
                                   workers=self.workers)
        with open(path, "wb") as f:
            pickle.dump(results, f)
        self._fresh.add(key)
        return results

    def note(self, key, value):
        self.numbers[key] = value

    def save_numbers(self):
        with open(os.path.join(RESULTS, "key_numbers.json"), "w") as f:
            json.dump(self.numbers, f, indent=1, default=float)
