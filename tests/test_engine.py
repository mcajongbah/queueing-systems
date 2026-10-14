"""Short tests demonstrating event handling and customer conservation.

Run with:  python -m pytest tests -q
"""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from simulator.engine import FutureEventList, PRIO_ARRIVAL, PRIO_DEPARTURE, PRIO_VACATION_END
from simulator.distributions import Deterministic, Exponential, TwoPoint
from simulator.system import ClassSpec, SystemSpec, QueueingSystem
from simulator import theory


def det_system(interarrival, service, servers=1, capacity=None, vacation=None, first=None):
    cls = ClassSpec(Deterministic(interarrival), Deterministic(service), first_arrival=first)
    return SystemSpec([cls], servers=servers, capacity=capacity, vacation=vacation)


def test_fel_orders_by_time_then_priority():
    fel = FutureEventList()
    fel.schedule(5.0, PRIO_ARRIVAL, "arrival", 0)
    fel.schedule(5.0, PRIO_DEPARTURE, "departure", 1)
    fel.schedule(5.0, PRIO_VACATION_END, "vacation_end", 2)
    fel.schedule(1.0, PRIO_ARRIVAL, "arrival", 3)
    kinds = [fel.pop()[3] for _ in range(4)]
    assert kinds == ["arrival", "departure", "vacation_end", "arrival"]


def test_departure_before_arrival_at_same_time():
    # arrivals at 1, 2, 3, ... with service exactly 1 s: departure of packet k and arrival of
    # packet k+1 coincide; the departure must go first so nobody ever waits.
    spec = det_system(1.0, 1.0, first=1.0)
    sim = QueueingSystem(spec, seed=0, t_warm=0.0, t_obs=20.0)
    res = sim.run()
    m = res["metrics"]
    assert m["W"] == 0.0
    assert m["T"] == pytest.approx(1.0)
    assert m["P_Q"] == 0.0
    assert m["conservation_ok"]


def test_arrival_does_not_restart_service():
    # single server, service 2 s, arrivals every 0.5 s starting at t=0.5
    spec = det_system(0.5, 2.0, first=0.5)
    sim = QueueingSystem(spec, seed=0, t_warm=0.0, t_obs=3.0, keep_records=True)
    res = sim.run()
    rec = res["records"]
    order = np.argsort(rec["arrival"])
    dep = rec["depart"][order]
    start = rec["start"][order]
    # first packet: starts 0.5, departs 2.5 regardless of the arrivals at 1.0, 1.5, 2.0
    assert start[0] == pytest.approx(0.5)
    assert dep[0] == pytest.approx(2.5)
    # second packet starts exactly when the first departs (FCFS, no restart)
    assert start[1] == pytest.approx(2.5)
    assert dep[1] == pytest.approx(4.5)


def test_deterministic_waiting_and_little():
    # D/D/1 with interarrival 1 s and service 0.6 s: nobody waits, N = 0.6 = lambda * T
    spec = det_system(1.0, 0.6, first=1.0)
    sim = QueueingSystem(spec, seed=0, t_warm=100.0, t_obs=1000.0)
    m = sim.run()["metrics"]
    assert m["W"] == 0.0
    assert m["T"] == pytest.approx(0.6)
    assert m["N"] == pytest.approx(0.6, rel=1e-6)
    assert m["frac_empty"] == pytest.approx(0.4, rel=1e-6)
    assert m["little_N_from_T"] == pytest.approx(m["N"], rel=1e-3)
    assert m["prob_sum"] == pytest.approx(1.0)
    assert m["conservation_ok"]


def test_multiple_servers_each_have_own_departure():
    # two servers, interarrival 1 s, service 1.5 s: alternating servers, nobody waits
    spec = det_system(1.0, 1.5, servers=2, first=1.0)
    # window [10, 70): busy time = 0.5 (arrival at 9) + 59 * 1.5 + 1.0 (arrival at 69) = 90 s
    m = QueueingSystem(spec, seed=0, t_warm=10.0, t_obs=60.0).run()["metrics"]
    assert m["W"] == 0.0
    assert m["busy_mean"] == pytest.approx(1.5, rel=1e-9)
    # same arrivals with one server would queue
    spec1 = det_system(1.0, 1.5, servers=1, first=1.0)
    m1 = QueueingSystem(spec1, seed=0, t_warm=0.0, t_obs=50.0).run()["metrics"]
    assert m1["W"] > 0.0
    assert m1["conservation_ok"] and m["conservation_ok"]


def test_loss_system_blocks_when_full():
    # capacity 1 (no waiting room), interarrival 1 s, service 1.5 s: every other arrival is lost
    spec = det_system(1.0, 1.5, servers=1, capacity=1, first=1.0)
    m = QueueingSystem(spec, seed=0, t_warm=0.0, t_obs=100.0).run()["metrics"]
    assert m["P_block"] == pytest.approx(0.5, abs=0.02)
    assert m["admitted"] + m["blocked"] == m["offered"]
    assert m["W"] == 0.0
    assert m["conservation_ok"]


def test_infinite_servers_never_queue():
    cls = ClassSpec(Exponential(50.0), Exponential(10.0))
    spec = SystemSpec([cls], servers=None)
    m = QueueingSystem(spec, seed=3, t_warm=10.0, t_obs=100.0).run()["metrics"]
    assert m["W"] == 0.0 and m["N_Q"] == 0.0 and m["P_Q"] == 0.0
    assert m["N"] == pytest.approx(5.0, rel=0.1)
    assert m["conservation_ok"]


def test_vacation_slots_align_and_never_interrupted():
    # deterministic V = X = 1 ms with a vacation at time zero: all service starts on 1 ms boundaries
    cls = ClassSpec(Exponential(400.0), Deterministic(1e-3))
    spec = SystemSpec([cls], servers=1, vacation=Deterministic(1e-3), start_with_vacation=True)
    res = QueueingSystem(spec, seed=1, t_warm=0.1, t_obs=1.0, keep_records=True).run()
    start = res["records"]["start"]
    frac = start / 1e-3
    assert np.max(np.abs(frac - np.round(frac))) < 1e-6
    m = res["metrics"]
    assert m["frac_serving"] + m["frac_vacation"] == pytest.approx(1.0, abs=1e-9)
    assert m["conservation_ok"]


def test_window_restriction_and_conservation_mm1():
    cls = ClassSpec(Exponential(800.0), Exponential(1000.0))
    spec = SystemSpec([cls], servers=1)
    sim = QueueingSystem(spec, seed=7, t_warm=1.0, t_obs=5.0)
    res = sim.run()
    m = res["metrics"]
    assert m["conservation_ok"]
    assert m["total_admitted"] - m["total_departed"] == m["final_occupancy"]
    assert m["offered"] == m["admitted"]                      # infinite buffer: nothing blocked
    assert m["n_delay_samples"] == m["admitted"]               # every window arrival followed to departure
    assert abs(m["prob_sum"] - 1.0) < 1e-9
    assert m["admitted_rate"] == pytest.approx(800.0, rel=0.05)
    th = theory.mm1(800.0, 1000.0)
    assert m["T"] == pytest.approx(th["T"], rel=0.15)


def test_distribution_moments():
    tp = TwoPoint(0.25e-3, 0.8, 4e-3)
    assert tp.mean == pytest.approx(1e-3)
    assert tp.second_moment == pytest.approx(3.25e-6)
    assert Exponential(1000.0).second_moment == pytest.approx(2e-6)
    assert Deterministic(1e-3).second_moment == pytest.approx(1e-6)
    rng = np.random.default_rng(0)
    x = tp.sample(rng, 200_000)
    assert x.mean() == pytest.approx(1e-3, rel=0.02)
    assert (x ** 2).mean() == pytest.approx(3.25e-6, rel=0.03)


def test_theory_anchors():
    assert theory.erlang_b(3.0, 3) == pytest.approx(0.346154, rel=1e-5)
    c = theory.erlang_c(1600.0, 1000.0, 2)
    assert c["P_Q"] == pytest.approx(0.71111, rel=1e-4)
    tc = theory.two_class_loss(2, 2, 1, 1, 3)
    assert tc["P_block_c1"] == pytest.approx(0.210526, rel=1e-5)
    assert sum(tc["pi"].values()) == pytest.approx(1.0)
    v = theory.mg1_vacation(400.0, 1e-3, 1e-6, 1e-3, 1e-6)
    assert v["W"] == pytest.approx(0.8333e-3, rel=1e-3)
