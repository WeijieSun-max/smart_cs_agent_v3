from __future__ import annotations

from typing import Any

from domain.customer_service_agent.interfaces.i_risk_checker import IRiskChecker


class RuleBasedRiskChecker(IRiskChecker):
    def check(self, user_id: str, action: str, amount: float = 0.0) -> dict[str, Any]:
        risk_level = "low"
        if amount > 50000:
            risk_level = "high"
        elif amount > 10000:
            risk_level = "medium"
        return {
            "user_id": user_id,
            "action": action,
            "amount": amount,
            "risk_level": risk_level,
            "requires_manual_review": risk_level == "high",
        }
