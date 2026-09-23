"""业务服务对外返回的电信用量画像和套餐比较模型。"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel


class UsageProfile(BaseModel):
    """基于最近账期计算的线路用量画像。

    `projected_*` 是当前未结束账期的线性外推，`recommended_*` 在峰值基础上
    增加安全余量。调用方必须结合 `data_quality` 展示置信度，不能把样本不足
    时的数值描述成确定预测。
    """

    line_id: str
    current_cycle_start: str | None = None
    current_cycle_end: str | None = None
    current_included_data_mb: int = 0
    current_used_data_mb: int = 0
    current_refueled_data_mb: int = 0
    current_remaining_data_mb: int = 0
    current_included_voice_minutes: int = 0
    current_used_voice_minutes: int = 0
    current_remaining_voice_minutes: int = 0
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
    """一个候选套餐相对用户预测用量的确定性比较结果。

    `dominated` 表示存在价格不高且数据、语音余量均不低的另一方案，便于
    Agent 排除帕累托劣势套餐，而不是让模型自行进行价格计算。
    """

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
