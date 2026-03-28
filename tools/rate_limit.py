from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    capacity: int
    refill_per_sec: float
    tokens: float
    last_refill_s: float

    @classmethod
    def per_minute(cls, *, limit: int) -> "TokenBucket":
        limit = max(1, limit)
        return cls(
            capacity=limit,
            refill_per_sec=limit / 60.0,
            tokens=float(limit),
            last_refill_s=time.monotonic(),
        )

    def consume(self, *, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill_s
        self.last_refill_s = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

