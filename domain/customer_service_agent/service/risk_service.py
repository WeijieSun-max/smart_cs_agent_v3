from __future__ import annotations

from typing import Any, Optional
import math

from domain.customer_service_agent.interfaces.i_risk_checker import IRiskChecker
from pkg.telemetry import traced_dependency
from pkg.exceptions.exception import ToolValidationError


class RiskService:
    def __init__(self, risk_checker: IRiskChecker):
        self.risk_checker = risk_checker

    @traced_dependency("risk.check", "tool")
    def check(self, user_id: str, action: str, amount: float = 0.0) -> dict[str, Any]:
        if not action.strip() or len(action) > 128 or not math.isfinite(amount) or amount < 0:
            raise ToolValidationError()
        return self.risk_checker.check(user_id=user_id, action=action, amount=amount)


instance: Optional[RiskService] = None


def initialize_service(risk_checker: IRiskChecker) -> None:
    global instance
    instance = RiskService(risk_checker)


def get_service() -> RiskService:
    if instance is None:
        raise RuntimeError("Risk service is not initialized")
    return instance
