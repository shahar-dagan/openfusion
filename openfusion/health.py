"""Provider health monitoring and latency tracking.

Passively observes every upstream call (success/failure + elapsed time) and
maintains per-provider health state and rolling latency percentiles. The router
and fallback logic consult this to avoid degraded or down providers.

Health state machine per provider:
  HEALTHY  → any recent success keeps the provider healthy
  DEGRADED → consecutive failures exceed _DEGRADED_THRESHOLD
  DOWN     → consecutive failures exceed _DOWN_THRESHOLD

Recovery: a single success resets consecutive_failures to 0 → HEALTHY.
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProviderStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


# Consecutive failures before marking a provider degraded / down.
_DEGRADED_THRESHOLD = 3
_DOWN_THRESHOLD = 8

# Rolling window size for latency percentile calculations.
_LATENCY_WINDOW = 100


@dataclass
class _ProviderBucket:
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    _latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))

    def record_success(self, latency_ms: float) -> None:
        self.consecutive_failures = 0
        self.total_requests += 1
        self._latency_ms.append(latency_ms)

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_requests += 1
        self.total_failures += 1

    @property
    def status(self) -> ProviderStatus:
        if self.consecutive_failures >= _DOWN_THRESHOLD:
            return ProviderStatus.DOWN
        if self.consecutive_failures >= _DEGRADED_THRESHOLD:
            return ProviderStatus.DEGRADED
        return ProviderStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        return self.status != ProviderStatus.DOWN

    def p50_ms(self) -> float | None:
        if len(self._latency_ms) < 5:
            return None
        return statistics.median(self._latency_ms)

    def p95_ms(self) -> float | None:
        samples = list(self._latency_ms)
        if len(samples) < 5:
            return None
        samples.sort()
        idx = max(0, int(len(samples) * 0.95) - 1)
        return samples[idx]


class HealthMonitor:
    """Thread-safe passive health tracker for upstream providers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, _ProviderBucket] = {}

    def _bucket(self, provider_id: str) -> _ProviderBucket:
        if provider_id not in self._buckets:
            self._buckets[provider_id] = _ProviderBucket()
        return self._buckets[provider_id]

    def record_success(self, provider_id: str, latency_ms: float) -> None:
        """Record a successful upstream call."""
        with self._lock:
            self._bucket(provider_id).record_success(latency_ms)

    def record_failure(self, provider_id: str) -> None:
        """Record a failed upstream call (HTTP error or network exception)."""
        with self._lock:
            self._bucket(provider_id).record_failure()

    def status(self, provider_id: str) -> ProviderStatus:
        with self._lock:
            b = self._buckets.get(provider_id)
        if b is None:
            return ProviderStatus.HEALTHY
        return b.status

    def is_available(self, provider_id: str) -> bool:
        return self.status(provider_id) != ProviderStatus.DOWN

    def p50_ms(self, provider_id: str) -> float | None:
        with self._lock:
            b = self._buckets.get(provider_id)
        return b.p50_ms() if b else None

    def p95_ms(self, provider_id: str) -> float | None:
        with self._lock:
            b = self._buckets.get(provider_id)
        return b.p95_ms() if b else None

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable summary for /v1/metrics or /v1/health."""
        with self._lock:
            buckets = dict(self._buckets)
        return {
            provider_id: {
                "status": b.status.value,
                "consecutive_failures": b.consecutive_failures,
                "total_requests": b.total_requests,
                "total_failures": b.total_failures,
                "p50_ms": round(p50, 1) if (p50 := b.p50_ms()) is not None else None,
                "p95_ms": round(p95, 1) if (p95 := b.p95_ms()) is not None else None,
            }
            for provider_id, b in sorted(buckets.items())
        }

    def available_providers(self) -> list[str]:
        """Return provider ids not currently marked DOWN."""
        with self._lock:
            return [pid for pid, b in self._buckets.items() if b.is_available]


# Module-level singleton — imported by upstream.py and server.py.
HEALTH = HealthMonitor()
