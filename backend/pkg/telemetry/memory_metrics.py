from __future__ import annotations

from collections import defaultdict
from threading import Lock


class MemoryMetrics:
    """Small in-process aggregate registry; labels are fixed and values are numeric only."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._packets = {
            "count": 0,
            "candidate_total": 0,
            "hydrated_total": 0,
            "filtered_total": 0,
            "injected_total": 0,
            "token_total": 0,
            "fallback_count": 0,
            "latency_ms_total": 0.0,
            "latency_ms_max": 0.0,
        }
        self._workers: dict[str, int] = defaultdict(int)
        self._extraction: dict[str, int] = defaultdict(int)

    def record_packet(
        self,
        *,
        candidates: int,
        hydrated: int,
        filtered: int,
        injected: int,
        tokens: int,
        latency_ms: float,
        fallback: bool,
    ) -> None:
        with self._lock:
            self._packets["count"] += 1
            self._packets["candidate_total"] += max(0, candidates)
            self._packets["hydrated_total"] += max(0, hydrated)
            self._packets["filtered_total"] += max(0, filtered)
            self._packets["injected_total"] += max(0, injected)
            self._packets["token_total"] += max(0, tokens)
            self._packets["fallback_count"] += int(fallback)
            self._packets["latency_ms_total"] += max(0.0, latency_ms)
            self._packets["latency_ms_max"] = max(
                self._packets["latency_ms_max"], max(0.0, latency_ms)
            )

    def record_worker(self, worker: str, outcome: str, count: int = 1) -> None:
        allowed_workers = {"extraction", "index"}
        allowed_outcomes = {"claimed", "completed", "retry", "dead", "loop_error"}
        if worker not in allowed_workers or outcome not in allowed_outcomes:
            return
        with self._lock:
            self._workers[f"{worker}.{outcome}"] += max(0, int(count))

    def record_extraction(self, stage: str, count: int) -> None:
        allowed = {"proposed", "accepted", "rejected", "created", "merged", "superseded"}
        if stage not in allowed:
            return
        with self._lock:
            self._extraction[stage] += max(0, int(count))

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            return {
                "packets": dict(self._packets),
                "workers": dict(sorted(self._workers.items())),
                "extraction": dict(sorted(self._extraction.items())),
            }

    def reset(self) -> None:
        with self._lock:
            for key in self._packets:
                self._packets[key] = 0.0 if key.startswith("latency_ms") else 0
            self._workers.clear()
            self._extraction.clear()


memory_metrics = MemoryMetrics()


def memory_metrics_snapshot() -> dict[str, dict[str, int | float]]:
    return memory_metrics.snapshot()
