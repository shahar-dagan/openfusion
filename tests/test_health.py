"""Tests for provider health monitoring and latency tracking."""

from __future__ import annotations

import threading

from openfusion.health import (
    _DEGRADED_THRESHOLD,
    _DOWN_THRESHOLD,
    HealthMonitor,
    ProviderStatus,
)
from openfusion.upstream import _provider_id_from_url

# ---------------------------------------------------------------------------
# _provider_id_from_url
# ---------------------------------------------------------------------------


def test_provider_id_openai() -> None:
    assert _provider_id_from_url("https://api.openai.com/v1") == "openai"


def test_provider_id_anthropic() -> None:
    assert _provider_id_from_url("https://api.anthropic.com/v1") == "anthropic"


def test_provider_id_groq() -> None:
    assert _provider_id_from_url("https://api.groq.com/openai/v1") == "groq"


def test_provider_id_openrouter() -> None:
    assert _provider_id_from_url("https://openrouter.ai/api/v1") == "openrouter"


def test_provider_id_localhost() -> None:
    pid = _provider_id_from_url("http://localhost:8080/v1")
    assert pid  # just check something is returned


# ---------------------------------------------------------------------------
# HealthMonitor — status transitions
# ---------------------------------------------------------------------------


def test_initial_status_is_healthy() -> None:
    mon = HealthMonitor()
    assert mon.status("openai") == ProviderStatus.HEALTHY


def test_single_failure_stays_healthy() -> None:
    mon = HealthMonitor()
    mon.record_failure("openai")
    assert mon.status("openai") == ProviderStatus.HEALTHY


def test_degraded_after_threshold_failures() -> None:
    mon = HealthMonitor()
    for _ in range(_DEGRADED_THRESHOLD):
        mon.record_failure("openai")
    assert mon.status("openai") == ProviderStatus.DEGRADED


def test_down_after_threshold_failures() -> None:
    mon = HealthMonitor()
    for _ in range(_DOWN_THRESHOLD):
        mon.record_failure("openai")
    assert mon.status("openai") == ProviderStatus.DOWN


def test_success_resets_to_healthy() -> None:
    mon = HealthMonitor()
    for _ in range(_DOWN_THRESHOLD):
        mon.record_failure("openai")
    assert mon.status("openai") == ProviderStatus.DOWN
    mon.record_success("openai", 100.0)
    assert mon.status("openai") == ProviderStatus.HEALTHY


def test_is_available_false_when_down() -> None:
    mon = HealthMonitor()
    for _ in range(_DOWN_THRESHOLD):
        mon.record_failure("openai")
    assert mon.is_available("openai") is False


def test_is_available_true_when_degraded() -> None:
    mon = HealthMonitor()
    for _ in range(_DEGRADED_THRESHOLD):
        mon.record_failure("openai")
    assert mon.is_available("openai") is True


def test_different_providers_tracked_independently() -> None:
    mon = HealthMonitor()
    for _ in range(_DOWN_THRESHOLD):
        mon.record_failure("openai")
    mon.record_success("anthropic", 50.0)
    assert mon.status("openai") == ProviderStatus.DOWN
    assert mon.status("anthropic") == ProviderStatus.HEALTHY


# ---------------------------------------------------------------------------
# HealthMonitor — latency percentiles
# ---------------------------------------------------------------------------


def test_p50_none_before_min_samples() -> None:
    mon = HealthMonitor()
    for i in range(4):
        mon.record_success("openai", float(i * 10))
    assert mon.p50_ms("openai") is None


def test_p50_available_after_min_samples() -> None:
    mon = HealthMonitor()
    for i in range(10):
        mon.record_success("openai", float(i * 10))
    assert mon.p50_ms("openai") is not None


def test_p95_higher_than_p50() -> None:
    mon = HealthMonitor()
    latencies = [10.0] * 90 + [1000.0] * 10  # 10% outliers
    for lat in latencies:
        mon.record_success("openai", lat)
    p50 = mon.p50_ms("openai")
    p95 = mon.p95_ms("openai")
    assert p50 is not None and p95 is not None
    assert p95 > p50


def test_p50_unknown_provider_returns_none() -> None:
    mon = HealthMonitor()
    assert mon.p50_ms("noprovider") is None


# ---------------------------------------------------------------------------
# HealthMonitor — snapshot
# ---------------------------------------------------------------------------


def test_snapshot_empty() -> None:
    mon = HealthMonitor()
    assert mon.snapshot() == {}


def test_snapshot_contains_provider_data() -> None:
    mon = HealthMonitor()
    mon.record_success("openai", 120.0)
    snap = mon.snapshot()
    assert "openai" in snap
    entry = snap["openai"]
    assert entry["status"] == "healthy"
    assert entry["total_requests"] == 1
    assert entry["total_failures"] == 0


def test_snapshot_includes_failure_count() -> None:
    mon = HealthMonitor()
    mon.record_failure("groq")
    snap = mon.snapshot()
    assert snap["groq"]["total_failures"] == 1


# ---------------------------------------------------------------------------
# HealthMonitor — available_providers
# ---------------------------------------------------------------------------


def test_available_providers_excludes_down() -> None:
    mon = HealthMonitor()
    mon.record_success("openai", 100.0)
    for _ in range(_DOWN_THRESHOLD):
        mon.record_failure("groq")
    available = mon.available_providers()
    assert "openai" in available
    assert "groq" not in available


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_records_are_safe() -> None:
    mon = HealthMonitor()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for i in range(50):
                if i % 5 == 0:
                    mon.record_failure("openai")
                else:
                    mon.record_success("openai", float(i))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    snap = mon.snapshot()
    assert snap["openai"]["total_requests"] == 400
