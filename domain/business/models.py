from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel


class UsageProfile(BaseModel):
    line_id: str
    usable_cycles: int
    analysis_period: str
    data_quality: Literal["insufficient", "low_confidence", "normal_confidence"]
    average_data_mb: int
    peak_data_mb: int
    projected_data_mb: int
    recommended_data_mb: int
    average_voice_minutes: int
    peak_voice_minutes: int
    projected_voice_minutes: int
    recommended_voice_minutes: int
    refuel_count: int = 0


class PlanComparison(BaseModel):
    plan_id: str
    name: str
    eligible: bool
    monthly_price: Decimal
    expected_monthly_cost: Decimal
    meets_data_need: bool
    meets_voice_need: bool
    data_buffer_mb: int
    voice_buffer_minutes: int
    dominated: bool = False
