from __future__ import annotations

from domain.shared.llm.llm_service import ModelProfile, initialize_llm_profiles
from infra.llm.openai_compatible_client import create_profile_chat_model
from pkg.config.settings import get_settings
from pkg.log.logger import get_logger

logger = get_logger()


def initialize_llm() -> None:
    settings = get_settings()
    default_model = settings.qwen_model
    profile_models = {
        "default": default_model,
        "supervisor": settings.qwen_supervisor_model or default_model,
        "knowledge_agent": settings.qwen_knowledge_agent_model or settings.qwen_response_writer_model or default_model,
        "telecom_agent": settings.qwen_telecom_agent_model or default_model,
        "retail_agent": settings.qwen_retail_agent_model or default_model,
        "response_writer": settings.qwen_response_writer_model or default_model,
        "safety_guard": settings.qwen_safety_guard_model or default_model,
        "memory": settings.qwen_memory_model or default_model,
    }
    profile_base_urls = {
        "default": settings.qwen_base_url,
        "supervisor": settings.qwen_supervisor_base_url or settings.qwen_base_url,
        "knowledge_agent": settings.qwen_knowledge_agent_base_url or settings.qwen_response_writer_base_url or settings.qwen_base_url,
        "telecom_agent": settings.qwen_telecom_agent_base_url or settings.qwen_base_url,
        "retail_agent": settings.qwen_retail_agent_base_url or settings.qwen_base_url,
        "response_writer": settings.qwen_response_writer_base_url or settings.qwen_base_url,
        "safety_guard": settings.qwen_safety_guard_base_url or settings.qwen_base_url,
        "memory": settings.qwen_memory_base_url or settings.qwen_base_url,
    }
    profiles = {
        name: ModelProfile(
            name=name,
            model=model,
            base_url=profile_base_urls[name],
            temperature=0.2 if name in {"response_writer", "knowledge_agent", "telecom_agent", "retail_agent"} else 0.0,
            max_tokens=1800 if name == "response_writer" else 1200,
            timeout_seconds=45.0 if name in {"supervisor", "knowledge_agent", "telecom_agent", "retail_agent"} else 30.0,
        )
        for name, model in profile_models.items()
    }
    clients = {name: create_profile_chat_model(settings, profile) for name, profile in profiles.items()}
    run_prefixes = {
        "supervisor.": "supervisor",
        "knowledge.": "knowledge_agent",
        "rag.": "response_writer",
        "telecom.": "telecom_agent",
        "retail.": "retail_agent",
        "response.": "response_writer",
        "fallback.": "response_writer",
        "compliance.": "safety_guard",
        "memory.": "memory",
    }
    initialize_llm_profiles(
        clients,
        run_prefixes=run_prefixes,
        max_concurrency=settings.llm_max_concurrency,
        queue_capacity=settings.llm_queue_capacity,
        queue_timeout_seconds=settings.llm_queue_timeout_seconds,
    )
    resolved = {
        name: (profile.model or default_model)
        for name, profile in profiles.items()
    }
    logger.info("LLM module initialized default_model={} profiles={}", default_model, resolved)
    logger.info("LLM run-prefix routing={}", run_prefixes)
