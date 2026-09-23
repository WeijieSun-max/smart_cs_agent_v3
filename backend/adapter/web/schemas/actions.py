from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionProposalRequest(BaseModel):
    user_id: str = Field(min_length=1,max_length=128,pattern=r"^[A-Za-z0-9._-]+$")
    session_id: str = Field(min_length=1,max_length=128,pattern=r"^[A-Za-z0-9._:-]+$")
    turn_id: str = Field(min_length=1,max_length=64,pattern=r"^[A-Za-z0-9_-]+$")
    tool_name: str = Field(min_length=1,max_length=128,pattern=r"^[a-z0-9_]+$")
    arguments: dict[str,Any]


class ActionDecisionRequest(BaseModel):
    user_id: str = Field(min_length=1,max_length=128,pattern=r"^[A-Za-z0-9._-]+$")
    session_id: str = Field(min_length=1,max_length=128,pattern=r"^[A-Za-z0-9._:-]+$")
    turn_id: str = Field(min_length=1,max_length=64,pattern=r"^[A-Za-z0-9_-]+$")
    decision: Literal["confirm","reject"]
