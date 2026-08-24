from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IRiskChecker(ABC):
    @abstractmethod
    def check(self, user_id: str, action: str, amount: float = 0.0) -> dict[str, Any]:
        pass
