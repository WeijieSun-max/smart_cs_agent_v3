from __future__ import annotations

from langchain_openai import ChatOpenAI

from pkg.config.settings import Settings
from domain.shared.llm.llm_service import ModelProfile


def create_profile_chat_model(settings: Settings, profile: ModelProfile) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.qwen_api_key or "missing-qwen-api-key",
        base_url=profile.base_url,
        model=profile.model,
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        timeout=profile.timeout_seconds,
        max_retries=profile.max_retries,
    )
