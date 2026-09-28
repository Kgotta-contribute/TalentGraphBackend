# -*- coding: utf-8 -*-
"""
Core Rate Limiter Module
Thread-safe and coroutine-safe multi-window sliding rate limiters for external APIs:
- Groq LLM: 20 RPM
- GitHub MCP: 20 RPM & 850 requests/hour

Ensures multiple concurrent users and agent workflows do not trigger upstream
HTTP 429 rate limit errors or exhaust provider quotas.
"""
import asyncio
import collections
import logging
import time
from typing import Any
from app.core.config import settings

logger = logging.getLogger("talent_agent.rate_limiter")


class RateLimitWindow:
    """Sliding time window for tracking and enforcing request counts."""

    def __init__(self, max_requests: int, window_seconds: float, label: str):
        self.max_requests = max(1, max_requests)
        self.window_seconds = float(window_seconds)
        self.label = label
        self.timestamps: collections.deque[float] = collections.deque()

    def cleanup(self, now: float) -> None:
        """Discard timestamps that have exited the rolling window."""
        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] <= cutoff:
            self.timestamps.popleft()

    def get_delay(self, now: float) -> float:
        """
        Calculates how many seconds to wait before a new request can be accepted.
        Returns 0.0 if capacity is currently available.
        """
        self.cleanup(now)
        if len(self.timestamps) < self.max_requests:
            return 0.0
        # Time until the oldest timestamp exits the window + 50ms buffer
        oldest = self.timestamps[0]
        delay = (oldest + self.window_seconds) - now
        return max(0.0, delay + 0.05)

    def record(self, now: float) -> None:
        """Record an admitted request timestamp."""
        self.timestamps.append(now)

    def stats(self, now: float) -> dict[str, Any]:
        self.cleanup(now)
        used = len(self.timestamps)
        remaining = max(0, self.max_requests - used)
        return {
            "label": self.label,
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "used": used,
            "remaining": remaining,
        }


class AsyncSlidingRateLimiter:
    """
    Multi-window sliding rate limiter.
    Supports multiple simultaneous constraints (e.g. 20 RPM AND 850 RPH).
    Uses asyncio.Lock to ensure thread/coroutine safety and FIFO fairness.
    """

    def __init__(self, limits: list[tuple[int, float, str]], name: str):
        """
        limits: list of (max_requests, window_seconds, label)
        e.g. [(20, 60.0, "20 RPM"), (850, 3600.0, "850 RPH")]
        """
        self.name = name
        self.windows = [
            RateLimitWindow(max_req, win_sec, lbl)
            for max_req, win_sec, lbl in limits
        ]
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Asynchronously blocks until all windows have capacity,
        then registers the request. Non-blocking to the rest of the event loop.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                delays = [w.get_delay(now) for w in self.windows]
                max_delay = max(delays, default=0.0)

                if max_delay <= 0.0:
                    # Capacity available under all windows
                    for w in self.windows:
                        w.record(now)
                    return

            # Sleep outside the lock so other requests/status queries are not blocked
            logger.info(
                f"[{self.name}] Rate limit reached. Queuing request for {max_delay:.2f}s..."
            )
            await asyncio.sleep(max_delay)

    def get_status(self) -> dict[str, Any]:
        """Returns live metrics about current utilization across all windows."""
        now = time.monotonic()
        delays = [w.get_delay(now) for w in self.windows]
        is_throttled = any(d > 0 for d in delays)
        return {
            "name": self.name,
            "is_throttled": is_throttled,
            "current_max_wait_seconds": round(max(delays, default=0.0), 2),
            "windows": [w.stats(now) for w in self.windows],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global Singletons
# ─────────────────────────────────────────────────────────────────────────────

# Groq LLM: 20 RPM
groq_rate_limiter = AsyncSlidingRateLimiter(
    limits=[(settings.groq_rpm_limit, 60.0, f"{settings.groq_rpm_limit} RPM")],
    name="Groq-LLM",
)

# GitHub MCP: 20 RPM & 850 requests/hour
github_rate_limiter = AsyncSlidingRateLimiter(
    limits=[
        (settings.github_rpm_limit, 60.0, f"{settings.github_rpm_limit} RPM"),
        (settings.github_rph_limit, 3600.0, f"{settings.github_rph_limit} RPH"),
    ],
    name="GitHub-MCP",
)
