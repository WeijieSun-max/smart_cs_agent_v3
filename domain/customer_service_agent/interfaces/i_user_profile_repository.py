from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IUserProfileRepository(ABC):
    @abstractmethod
    def get_profile(self, user_id: str) -> dict[str, Any]:
        pass
