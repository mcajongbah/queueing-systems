"""The reusable queueing system.

One class covers every model in the homework; the differences are expressed
only through the specification:

    servers    1 (M/M/1, M/G/1), m (M/M/m), None = infinitely many (M/M/inf)
    capacity   None = infinite buffer; K = at most K customers in the system,
               arrivals finding K customers are blocked and lost (M/M/m/m: K=m)
    classes    one or more independent arrival streams, each with its own
               interarrival and service-time distributions (two-class loss)
    vacation   None, or a distribution of vacation lengths (exhaustive service
               with repeated vacations; slotted M/D/1 when V = X deterministic)

State kept by the simulator: the clock, the number in system n (per class),
the number of busy servers, the FCFS waiting queue, the vacation flag and the
future-event list.  The packet in service keeps its scheduled departure event;
an arrival never touches it (no service restart, no resampling).
"""
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .engine import (FutureEventList, PRIO_ARRIVAL, PRIO_DEPARTURE, PRIO_VACATION_END,
                     EV_ARRIVAL, EV_DEPARTURE, EV_VACATION_END)
from .distributions import Distribution, Sampler
from .stats import WindowStats


@dataclass
class ClassSpec:
    interarrival: Distribution
    service: Distribution
    name: str = "c1"
    first_arrival: Optional[float] = None   # None: draw the first interarrival at t=0

    @property
    def rate(self):
        return 1.0 / self.interarrival.mean


@dataclass
class SystemSpec:
    classes: List[ClassSpec]
    servers: Optional[int] = 1          # None -> infinitely many servers
    capacity: Optional[int] = None      # None -> infinite waiting room
    vacation: Optional[Distribution] = None
    start_with_vacation: bool = True
    label: str = ""

    @property
    def total_rate(self):
        return sum(c.rate for c in self.classes)

    def describe(self):
        parts = [f"servers={'inf' if self.servers is None else self.servers}",
                 f"capacity={'inf' if self.capacity is None else self.capacity}"]
        for c in self.classes:
            parts.append(f"{c.name}: arrivals {c.interarrival.describe()}, service {c.service.describe()}")
        if self.vacation is not None:
            parts.append(f"vacation {self.vacation.describe()}")
        return "; ".join(parts)


class Customer:
    __slots__ = ("cls", "arrival", "start", "service", "in_window")

    def __init__(self, cls, arrival, in_window):
        self.cls = cls
        self.arrival = arrival
        self.start = None
        self.service = None
        self.in_window = in_window


class QueueingSystem:
    """Event-driven simulator for one replication of one configuration."""

    def __init__(self, spec: SystemSpec, seed, t_warm, t_obs, keep_records=False):
        self.spec = spec
        self.seed = seed
        self.t_warm = float(t_warm)
        self.t_obs = float(t_obs)
        self.keep_records = keep_records
        k = len(spec.classes)
        streams = np.random.SeedSequence(seed).spawn(2 * k + 1)
        self.arrival_gen = [Sampler(c.interarrival, streams[2 * i]) for i, c in enumerate(spec.classes)]
        self.service_gen = [Sampler(c.service, streams[2 * i + 1]) for i, c in enumerate(spec.classes)]
        self.vacation_gen = Sampler(spec.vacation, streams[-1]) if spec.vacation is not None else None
        self.servers = spec.servers
        self.capacity = spec.capacity
        # state
        self.clock = 0.0
        self.n = 0
        self.n_class = [0] * k
        self.busy = 0
        self.on_vacation = False
        self.queue = deque()
        self.fel = FutureEventList()
        self.total_admitted = 0
        self.total_departed = 0
        self.total_blocked = 0
        self.events_processed = 0
        self.stats = WindowStats(self.t_warm, self.t_warm + self.t_obs, n_classes=k,
                                 track_joint=(k > 1))

    # ----------------------------------------------------------------- run
    def run(self):
        spec = self.spec
        t1 = self.stats.t1
        for i, c in enumerate(spec.classes):
            t = c.first_arrival if c.first_arrival is not None else self.arrival_gen[i]()
            self.fel.schedule(t, PRIO_ARRIVAL, EV_ARRIVAL, i)
        if self.vacation_gen is not None and spec.start_with_vacation:
            self._start_vacation()
        fel = self.fel
        stats = self.stats
        while fel:
            time, _prio, _seq, kind, payload = fel.pop()
            if time >= t1 and stats.pending == 0:
                break
            self.clock = time
            self.events_processed += 1
            if kind == EV_ARRIVAL:
                self._arrival(payload)
            elif kind == EV_DEPARTURE:
                self._departure(payload)
            else:
                self._vacation_end()
        # close the time-average window with the state that held at the end
        stats.accumulate(max(self.clock, t1), self.n, self.busy, self.on_vacation, self.n_class)
        result = stats.summary(keep_records=self.keep_records)
        m = result["metrics"]
        m["seed"] = self.seed
        m["warmup_s"] = self.t_warm
        m["observation_s"] = self.t_obs
        m["total_admitted"] = self.total_admitted
        m["total_departed"] = self.total_departed
        m["total_blocked"] = self.total_blocked
        m["final_occupancy"] = self.n
        m["conservation_ok"] = (self.total_admitted - self.total_departed == self.n)
        m["end_time_s"] = self.clock
        m["events"] = self.events_processed
        return result

    # ------------------------------------------------------------ handlers
    def _arrival(self, cls):
        now = self.clock
        # next arrival of this class (independent of everything else)
        self.fel.schedule(now + self.arrival_gen[cls](), PRIO_ARRIVAL, EV_ARRIVAL, cls)
        stats = self.stats
        n_seen = self.n                       # state seen BEFORE admission / occupancy change
        in_window = stats.t0 <= now < stats.t1
        if in_window:
            stats.record_offered(cls, n_seen)
        if self.capacity is not None and self.n >= self.capacity:
            self.total_blocked += 1
            if in_window:
                stats.record_blocked(cls)
            return
        stats.accumulate(now, self.n, self.busy, self.on_vacation, self.n_class)
        self.n += 1
        self.n_class[cls] += 1
        self.total_admitted += 1
        cust = Customer(cls, now, in_window)
        immediate = (not self.on_vacation) and (self.servers is None or self.busy < self.servers)
        if in_window:
            stats.record_admitted(cls, n_seen, self.n, immediate)
        if immediate:
            self._start_service(cust)
        else:
            self.queue.append(cust)

    def _start_service(self, cust):
        x = self.service_gen[cust.cls]()
        cust.start = self.clock
        cust.service = x
        self.busy += 1
        # the departure is fixed now and never rescheduled
        self.fel.schedule(self.clock + x, PRIO_DEPARTURE, EV_DEPARTURE, cust)

    def _departure(self, cust):
        now = self.clock
        self.stats.accumulate(now, self.n, self.busy, self.on_vacation, self.n_class)
        self.n -= 1
        self.n_class[cust.cls] -= 1
        self.busy -= 1
        self.total_departed += 1
        if cust.in_window:
            self.stats.record_departure(cust, now)
        if self.queue:
            self._start_service(self.queue.popleft())
        elif self.vacation_gen is not None and self.n == 0:
            self._start_vacation()

    def _start_vacation(self):
        self.stats.accumulate(self.clock, self.n, self.busy, self.on_vacation, self.n_class)
        self.on_vacation = True
        self.fel.schedule(self.clock + self.vacation_gen(), PRIO_VACATION_END, EV_VACATION_END, None)

    def _vacation_end(self):
        self.stats.accumulate(self.clock, self.n, self.busy, self.on_vacation, self.n_class)
        self.on_vacation = False
        if self.queue:
            self._start_service(self.queue.popleft())
        else:
            self._start_vacation()
