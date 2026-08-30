"""Framework-free state bookkeeping for VoXtream2 sink attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SinkAttentionConfig:
    """Match the cache rebuild policy used by upstream VoXtream2."""

    audio_window_size: int
    min_recent_window: int = 1
    tail_budget_padding: int = 1
    tail_window_divisor: int = 2

    def __post_init__(self) -> None:
        if self.audio_window_size < 2:
            raise ValueError("audio_window_size must be at least 2")
        if self.min_recent_window < 1:
            raise ValueError("min_recent_window must be positive")
        if self.tail_budget_padding < 0:
            raise ValueError("tail_budget_padding cannot be negative")
        if self.tail_window_divisor < 1:
            raise ValueError("tail_window_divisor must be positive")


class SinkAttentionHistory:
    """Keep the immutable prompt and the recent hidden-input tail.

    The temporal transformer cache cannot be shifted in place: cached keys
    contain position-dependent RoPE values and deeper layers depend on the
    retained context. Upstream VoXtream2 resets the cache and replays the
    prompt plus a bounded recent tail at new local positions. This class owns
    only that deterministic policy; the TensorRT runtime performs the replay.
    """

    def __init__(self, config: SinkAttentionConfig, prompt_hidden: np.ndarray) -> None:
        prompt_hidden = np.ascontiguousarray(prompt_hidden)
        if prompt_hidden.ndim != 3:
            raise ValueError("prompt_hidden must have shape (batch, sequence, hidden)")
        if prompt_hidden.shape[1] >= config.audio_window_size:
            raise ValueError("prompt must be shorter than the sink-attention window")
        self.config = config
        self.prompt_hidden = prompt_hidden.copy()
        self.prompt_length = int(prompt_hidden.shape[1])
        tail_budget = max(
            0,
            config.audio_window_size
            - self.prompt_length
            - config.tail_budget_padding,
        )
        self.tail_limit = min(
            config.audio_window_size // config.tail_window_divisor,
            tail_budget,
        )
        self.reset()

    def reset(self) -> None:
        self.position_offset = 0
        self._tail: list[np.ndarray] = []
        self.compactions = 0

    @property
    def tail_length(self) -> int:
        return len(self._tail)

    def local_position(self, global_position: int) -> int:
        local_position = int(global_position) - self.position_offset
        if local_position < 0:
            raise ValueError("global position precedes the current sink-attention segment")
        return local_position

    def needs_rebuild(self, global_position: int) -> bool:
        return self.local_position(global_position) >= self.config.audio_window_size

    def append(self, hidden: np.ndarray) -> None:
        hidden = np.ascontiguousarray(hidden)
        expected = (
            self.prompt_hidden.shape[0],
            1,
            self.prompt_hidden.shape[2],
        )
        if hidden.shape != expected:
            raise ValueError(f"sink tail hidden shape changed: {hidden.shape} != {expected}")
        if hidden.dtype != self.prompt_hidden.dtype:
            raise ValueError(
                "sink tail hidden dtype changed: "
                f"{hidden.dtype} != {self.prompt_hidden.dtype}"
            )
        if self.tail_limit <= 0:
            self._tail = []
            return
        self._tail.append(hidden.copy())
        if len(self._tail) > self.tail_limit:
            del self._tail[: len(self._tail) - self.tail_limit]

    def rebuild_sequence(self, global_position: int) -> np.ndarray:
        if not self.needs_rebuild(global_position):
            raise ValueError("sink-attention rebuild requested before the window is full")
        pieces = [self.prompt_hidden, *self._tail]
        hidden = np.ascontiguousarray(np.concatenate(pieces, axis=1))
        sequence_length = int(hidden.shape[1])
        if sequence_length >= self.config.audio_window_size:
            raise RuntimeError("sink-attention rebuild did not free a cache position")
        self.position_offset = int(global_position) - sequence_length
        if self.local_position(global_position) != sequence_length:
            raise RuntimeError("sink-attention local position invariant failed")
        self.compactions += 1
        return hidden
