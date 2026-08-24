from __future__ import annotations

import hashlib


def cohort_bucket(user_id: str,salt: str="smart-cs-canary-v1") -> int:
    return int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:8],16)%100


def in_canary(user_id: str,percentage: int) -> bool:
    return percentage>0 and cohort_bucket(user_id)<percentage
