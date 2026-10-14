"""
Reputation ledger — tracks PREDICTION ACCURACY, not task success. An agent
that always does great work but wildly mis-predicts its own token cost is
exactly the agent that breaks the auction's rate-limit math, so that's what
gets penalized here.

    accuracy = min(predicted, actual) / max(predicted, actual)

This is symmetric: over-predicting and under-predicting are punished
equally. That symmetry matters — the auction's bid_score formula already
disincentivizes over-predicting, but nothing about bid_score stops
under-predicting (in fact it inflates score). Reputation is what closes that
loophole, over time, for the "freeloader" strategy.

Also owns collusion detection: agents can endorse each other's predictions
(a legitimate way to vouch for a new agent), but a set of agents that only
ever endorse each other in a cycle is a collusion ring propping up
reputation without real accuracy — flag it.
"""
import itertools
import time
from dataclasses import dataclass, field
from threading import Lock

from . import config
from .event_log import event_log

_event_id_counter = itertools.count(1)


@dataclass
class ReputationEvent:
    id: int
    agent_id: str
    timestamp: float
    predicted_tokens: int
    actual_tokens: int
    accuracy: float
    delta: float
    new_reputation: float
    flagged_liar: bool
    dropped: bool
    newly_flagged: bool = False
    newly_dropped: bool = False


@dataclass
class AgentReputation:
    agent_id: str
    reputation: float = config.REPUTATION_START
    accuracy_history: list = field(default_factory=list)
    flagged_liar: bool = False
    dropped: bool = False
    calls: int = 0

    @property
    def avg_accuracy(self) -> float:
        if not self.accuracy_history:
            return 1.0
        return sum(self.accuracy_history) / len(self.accuracy_history)


class ReputationLedger:
    def __init__(self):
        self._lock = Lock()
        self._agents: dict[str, AgentReputation] = {}
        self.events: list[ReputationEvent] = []
        # Directed endorsement graph: endorsements[a] = set of agents a vouched for.
        self.endorsements: dict[str, set[str]] = {}

    def _get_or_create(self, agent_id: str) -> AgentReputation:
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentReputation(agent_id=agent_id)
        return self._agents[agent_id]

    def get(self, agent_id: str) -> float:
        with self._lock:
            return self._get_or_create(agent_id).reputation

    def is_active(self, agent_id: str) -> bool:
        with self._lock:
            a = self._agents.get(agent_id)
            return a is None or not a.dropped

    def record_prediction_outcome(self, agent_id: str, predicted_tokens: int, actual_tokens: int) -> ReputationEvent:
        with self._lock:
            agent = self._get_or_create(agent_id)
            p, a = max(1, predicted_tokens), max(1, actual_tokens)
            accuracy = min(p, a) / max(p, a)
            agent.accuracy_history.append(accuracy)
            agent.calls += 1

            delta = 0.0
            newly_flagged = False
            if accuracy > config.REPUTATION_ACCURACY_HIGH:
                delta = config.REPUTATION_BONUS
            elif accuracy < config.REPUTATION_ACCURACY_LOW:
                delta = -config.REPUTATION_PENALTY
                if not agent.flagged_liar:
                    agent.flagged_liar = True
                    newly_flagged = True

            agent.reputation = min(config.REPUTATION_MAX, max(config.REPUTATION_MIN, agent.reputation + delta))

            newly_dropped = False
            if agent.reputation < config.REPUTATION_DROP_THRESHOLD and not agent.dropped:
                agent.dropped = True
                newly_dropped = True

            ev = ReputationEvent(
                id=next(_event_id_counter),
                agent_id=agent_id,
                timestamp=time.time(),
                predicted_tokens=predicted_tokens,
                actual_tokens=actual_tokens,
                accuracy=accuracy,
                delta=delta,
                new_reputation=agent.reputation,
                flagged_liar=agent.flagged_liar,
                dropped=agent.dropped,
                newly_flagged=newly_flagged,
                newly_dropped=newly_dropped,
            )
            self.events.append(ev)

        if newly_flagged:
            event_log.add(
                "reputation",
                f"agent {agent_id} flagged: predicted {predicted_tokens}, used {actual_tokens}, "
                f"accuracy {accuracy:.0%} — reputation {delta:+.0f} -> {ev.new_reputation:.0f}",
            )
        elif delta > 0:
            event_log.add(
                "reputation",
                f"agent {agent_id} accurate prediction: predicted {predicted_tokens}, used {actual_tokens}, "
                f"accuracy {accuracy:.0%} — reputation {delta:+.0f} -> {ev.new_reputation:.0f}",
            )
        if newly_dropped:
            event_log.add(
                "reputation",
                f"agent {agent_id} DROPPED from swarm: reputation {ev.new_reputation:.0f} "
                f"< threshold {config.REPUTATION_DROP_THRESHOLD:.0f}",
            )
        return ev

    # --- Collusion detection ---

    def endorse(self, from_agent: str, to_agent: str):
        if from_agent == to_agent:
            return
        with self._lock:
            self.endorsements.setdefault(from_agent, set()).add(to_agent)
        event_log.add("reputation", f"agent {from_agent} endorsed agent {to_agent}")

    def detect_collusion_cycles(self) -> list[list[str]]:
        """DFS-based cycle detection over the endorsement graph. Any cycle
        (A endorses B endorses C endorses A, ...) is flagged — mutual
        vouching in a closed loop with no outside endorsements is the
        collusion-ring pattern, not a natural trust structure."""
        with self._lock:
            graph = {k: set(v) for k, v in self.endorsements.items()}

        cycles: list[list[str]] = []
        visited: set[str] = set()
        stack: list[str] = []
        on_stack: set[str] = set()

        def dfs(node: str):
            visited.add(node)
            stack.append(node)
            on_stack.add(node)
            for neighbor in graph.get(node, ()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in on_stack:
                    idx = stack.index(neighbor)
                    cycle = stack[idx:]
                    if len(cycle) >= 2:
                        normalized = sorted(cycle)
                        if normalized not in [sorted(c) for c in cycles]:
                            cycles.append(cycle)
            stack.pop()
            on_stack.discard(node)

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node)

        if cycles:
            for cycle in cycles:
                event_log.add(
                    "reputation",
                    f"collusion ring detected: {' -> '.join(cycle + [cycle[0]])} — "
                    f"flagging {len(cycle)} agents",
                )
                with self._lock:
                    for agent_id in cycle:
                        agent = self._get_or_create(agent_id)
                        agent.flagged_liar = True

        return cycles

    def snapshot(self) -> list[dict]:
        with self._lock:
            agents = list(self._agents.values())
        return [
            {
                "agent_id": a.agent_id,
                "reputation": round(a.reputation, 1),
                "avg_accuracy": round(a.avg_accuracy, 3),
                "calls": a.calls,
                "flagged_liar": a.flagged_liar,
                "dropped": a.dropped,
            }
            for a in sorted(agents, key=lambda x: x.reputation, reverse=True)
        ]


reputation_ledger = ReputationLedger()
