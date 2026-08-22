"""Dependency-free Real-Time Chunking request scheduler."""

from __future__ import annotations

from collections import deque


class RTCScheduler:
    """Track one in-flight replan and its latency in control steps."""

    def __init__(
        self,
        *,
        min_exec_horizon: int,
        initial_delay_steps: int,
        delay_buffer_size: int,
    ) -> None:
        if isinstance(min_exec_horizon, bool) or min_exec_horizon < 1:
            raise ValueError("min_exec_horizon must be positive")
        if isinstance(initial_delay_steps, bool) or initial_delay_steps < 0:
            raise ValueError("initial_delay_steps must be non-negative")
        if isinstance(delay_buffer_size, bool) or delay_buffer_size < 1:
            raise ValueError("delay_buffer_size must be positive")
        self.min_exec_horizon = int(min_exec_horizon)
        self._delays = deque([int(initial_delay_steps)], maxlen=int(delay_buffer_size))
        self._request_step: int | None = None

    @property
    def pending(self) -> bool:
        return self._request_step is not None

    @property
    def predicted_delay_steps(self) -> int:
        return max(self._delays)

    def should_request(self, chunk_index: int, chunk_length: int) -> bool:
        return not self.pending and self.min_exec_horizon <= int(chunk_index) < int(chunk_length)

    def mark_requested(self, episode_step: int) -> None:
        if self.pending:
            raise RuntimeError("an RTC request is already in flight")
        self._request_step = int(episode_step)

    def resolve(self, episode_step: int) -> int:
        if self._request_step is None:
            raise RuntimeError("no RTC request is in flight")
        observed = max(int(episode_step) - self._request_step, 0)
        self._delays.append(observed)
        self._request_step = None
        return observed
