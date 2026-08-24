from __future__ import annotations

from typing import Any, Optional

from domain.customer_service_agent.interfaces.i_user_profile_repository import IUserProfileRepository
from pkg.telemetry import traced_dependency


class UserProfileService:
    def __init__(self, repository: IUserProfileRepository):
        self.repository = repository

    @traced_dependency("user-profile.get", "tool")
    def get_profile(self, user_id: str) -> dict[str, Any]:
        return self.repository.get_profile(user_id)


instance: Optional[UserProfileService] = None


def initialize_service(repository: IUserProfileRepository) -> None:
    global instance
    instance = UserProfileService(repository)


def get_service() -> UserProfileService:
    if instance is None:
        raise RuntimeError("User profile service is not initialized")
    return instance
