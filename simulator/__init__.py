"""simulator: a small event-driven queueing simulator for ECE 533 Homework 2.

Modules
-------
engine         future-event list and event priorities (tie handling)
distributions  interarrival / service / vacation time generators
stats          window-restricted measurement counters
system         the reusable queueing system (arrival, departure, vacation handlers)
protocol       replication runner, confidence intervals, CSV helpers
theory         analytical reference formulas (kept separate from measurements)
"""
from .engine import FutureEventList, PRIO_DEPARTURE, PRIO_VACATION_END, PRIO_ARRIVAL
from .distributions import Exponential, Deterministic, TwoPoint, Sampler
from .system import ClassSpec, SystemSpec, QueueingSystem
from .stats import NBINS, bin_index
