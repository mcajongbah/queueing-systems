"""Future-event list (FEL).

The simulation clock always jumps directly to the time of the next scheduled
event; there are no fixed time steps.  Events are kept in a binary heap keyed by
(time, priority, sequence number).  The priority resolves ties between events
scheduled at exactly the same instant:

    departures (0)  before  vacation ends (1)  before  arrivals (2)

The sequence number makes the key unique, so two events of the same type at the
same time are processed in the order they were scheduled, and the payload is
never compared.
"""
import heapq
from itertools import count

PRIO_DEPARTURE = 0
PRIO_VACATION_END = 1
PRIO_ARRIVAL = 2

EV_ARRIVAL = "arrival"
EV_DEPARTURE = "departure"
EV_VACATION_END = "vacation_end"


class FutureEventList:
    __slots__ = ("_heap", "_seq")

    def __init__(self):
        self._heap = []
        self._seq = count()

    def schedule(self, time, priority, kind, payload=None):
        """Insert an event.  Returns the heap entry (useful in tests)."""
        entry = (time, priority, next(self._seq), kind, payload)
        heapq.heappush(self._heap, entry)
        return entry

    def pop(self):
        """Remove and return the earliest event as (time, priority, seq, kind, payload)."""
        return heapq.heappop(self._heap)

    def peek_time(self):
        return self._heap[0][0]

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)
