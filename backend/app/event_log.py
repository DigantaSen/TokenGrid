"""
The event log is half the demo: every mechanism that makes a decision
(auction clears a bid, reputation flags a liar, cache hits, swarm resizes...)
writes one line here explaining *why*. Judges read this to see causality,
not just outcomes.
"""
import itertools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

_event_id_counter = itertools.count(1)
MAX_EVENTS = 2000


@dataclass
class Event:
    id: int
    timestamp: float
    category: str  # "auction" | "reputation" | "cache" | "swarm" | "router" | "stress" | "system"
    message: str
    meta: dict = field(default_factory=dict)


class EventLog:
    def __init__(self):
        self._lock = Lock()
        self.events: list[Event] = []
        self.enabled = True

    @contextmanager
    def muted(self):
        """Offline experiments run the real mechanisms for thousands of
        rounds; logging every decision there would only cost time."""
        prev, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = prev

    def add(self, category: str, message: str, **meta) -> Event:
        ev = Event(id=next(_event_id_counter), timestamp=time.time(), category=category, message=message, meta=meta)
        if not self.enabled:
            return ev
        with self._lock:
            self.events.append(ev)
            if len(self.events) > MAX_EVENTS:
                self.events = self.events[-MAX_EVENTS:]
        return ev

    def recent(self, n: int = 50) -> list[Event]:
        with self._lock:
            return list(reversed(self.events[-n:]))

    def clear(self):
        with self._lock:
            self.events = []


event_log = EventLog()
