from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IOrderQuery(ABC):
    @abstractmethod
    def query_order(self, order_id: str = "", user_id: str = "") -> dict[str, Any]:
        pass
