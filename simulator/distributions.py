"""Random-variate generators for interarrival, service and vacation times.

Every distribution exposes its exact first two moments so that the analytical
references can be computed from the same objects that drive the simulator.
Sampling is done in blocks from an independent numpy Generator per stream
(arrivals, services and vacations never share a random stream).
"""
import numpy as np


class Distribution:
    name = "base"

    def sample(self, rng, size):
        raise NotImplementedError

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def second_moment(self):
        raise NotImplementedError

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2

    @property
    def rate(self):
        return 1.0 / self.mean

    def __repr__(self):
        return self.describe()


class Exponential(Distribution):
    name = "exponential"

    def __init__(self, rate):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)

    def sample(self, rng, size):
        return rng.exponential(1.0 / self._rate, size)

    @property
    def mean(self):
        return 1.0 / self._rate

    @property
    def second_moment(self):
        return 2.0 / self._rate ** 2

    def describe(self):
        return f"Exponential(rate={self._rate:g}/s)"


class Deterministic(Distribution):
    name = "deterministic"

    def __init__(self, value):
        if value <= 0:
            raise ValueError("value must be positive")
        self.value = float(value)

    def sample(self, rng, size):
        return np.full(size, self.value)

    @property
    def mean(self):
        return self.value

    @property
    def second_moment(self):
        return self.value ** 2

    def describe(self):
        return f"Deterministic({self.value:g} s)"


class TwoPoint(Distribution):
    """Takes value a with probability p and value b with probability 1-p."""
    name = "two_point"

    def __init__(self, a, p, b):
        self.values = np.array([float(a), float(b)])
        self.probs = np.array([float(p), 1.0 - float(p)])

    def sample(self, rng, size):
        return rng.choice(self.values, size=size, p=self.probs)

    @property
    def mean(self):
        return float(np.dot(self.probs, self.values))

    @property
    def second_moment(self):
        return float(np.dot(self.probs, self.values ** 2))

    def describe(self):
        return (f"TwoPoint({self.values[0]:g} s w.p. {self.probs[0]:g}, "
                f"{self.values[1]:g} s w.p. {self.probs[1]:g})")


class Sampler:
    """Block-buffered variate source bound to one independent random stream."""
    __slots__ = ("dist", "rng", "block", "_buf", "_i")

    def __init__(self, dist, seed_sequence, block=8192):
        self.dist = dist
        self.rng = np.random.default_rng(seed_sequence)
        self.block = block
        self._buf = []
        self._i = 0

    def __call__(self):
        if self._i >= len(self._buf):
            self._buf = self.dist.sample(self.rng, self.block).tolist()
            self._i = 0
        v = self._buf[self._i]
        self._i += 1
        return v


def parse_distribution(text):
    """Parse a CLI specification such as 'exp:1000', 'det:0.001' or 'two:0.00025,0.8,0.004'.

    exp:RATE         exponential with the given rate (per second)
    det:VALUE        deterministic value (seconds)
    two:A,P,B        value A with probability P, value B with probability 1-P (seconds)
    """
    kind, _, arg = text.partition(":")
    kind = kind.lower()
    if kind in ("exp", "exponential"):
        return Exponential(float(arg))
    if kind in ("det", "deterministic"):
        return Deterministic(float(arg))
    if kind in ("two", "two_point", "twopoint"):
        a, p, b = (float(x) for x in arg.split(","))
        return TwoPoint(a, p, b)
    raise ValueError(f"unknown distribution spec: {text!r}")
