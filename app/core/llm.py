# -*- coding: utf-8 -*-
"""
LangChain Unified LLM Factory Module
Provides multi-provider model switching (Groq, OpenAI, Anthropic, Ollama, etc.)
with rate limiting, fallback models, and native Pydantic structured output.
"""
import json
import logging
from typing import Type, TypeVar, Any
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.core.rate_limiter import groq_rate_limiter

logger = logging.getLogger("talent_agent.langchain_llm")

T = TypeVar("T", bound=BaseModel)

def get_models_for_provider(provider: str) -> list[str]:
    """Return fallback model list tailored to the active provider."""
    if provider == "openai":
        primary = settings.openai_model or "gpt-4o-mini"
        return list(dict.fromkeys([primary, "gpt-4o", "gpt-3.5-turbo"]))
    elif provider == "anthropic":
        primary = settings.anthropic_model or "claude-3-5-sonnet-20241022"
        return list(dict.fromkeys([primary, "claude-3-haiku-20240307"]))
    else:
        # Default: Groq
        # Exact sequence: groq_model → openai/gpt-oss-120b → openai/gpt-oss-20b → qwen/qwen3.8-27b
        candidates = [
            settings.groq_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]
        return list(dict.fromkeys([m for m in candidates if m]))



def get_chat_model(model_name: str | None = None, temperature: float = 0.1):
    """
    Universal LangChain ChatModel factory.
    Easily switch providers via LLM_PROVIDER in .env:
    - 'groq' (default, free & ultra-fast)
    - 'openai' (GPT-4o, GPT-4o-mini)
    - 'anthropic' (Claude 3.5 Sonnet)
    """
    provider = settings.llm_provider.lower().strip()

    if provider == "openai":
        selected_model = model_name or settings.openai_model
        return ChatOpenAI(
            model=selected_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=temperature,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        selected_model = model_name or settings.anthropic_model
        return ChatAnthropic(
            model=selected_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
        )
    else:
        # Default: Groq via OpenAI-compatible endpoint
        selected_model = model_name or settings.groq_model
        return ChatOpenAI(
            model=selected_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            temperature=temperature,
        )


async def invoke_structured_output(
    system_prompt: str,
    user_content: str,
    schema: Type[T],
    temperature: float = 0.1,
) -> T:
    """
    Invoke LLM with LangChain .with_structured_output(schema).
    Guarantees validated Pydantic schema return with automatic retry across fallback models.
    """
    provider = settings.llm_provider.lower().strip()
    if provider == "groq":
        await groq_rate_limiter.acquire()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    candidate_models = get_models_for_provider(provider)
    last_err = None
    for model_name in candidate_models:
        try:
            model = get_chat_model(model_name=model_name, temperature=temperature)
            structured_llm = model.with_structured_output(schema)
            result = await structured_llm.ainvoke(messages)
            if result:
                return result
        except Exception as e:
            last_err = e
            logger.warning(f"[LangChain {provider}] Model {model_name} failed: {e}. Trying fallback...")

    logger.error(f"[LangChain {provider}] All models failed. Last error: {last_err}")
    raise RuntimeError(f"LangChain invocation failed across all fallback models: {last_err}")


async def invoke_json(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.1,
) -> dict:
    """
    Invoke LLM with LangChain returning parsed JSON dict.
    Enforces json_object format, rate-limiting, and multi-model fallback.
    """
    provider = settings.llm_provider.lower().strip()
    if provider == "groq":
        await groq_rate_limiter.acquire()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    candidate_models = get_models_for_provider(provider)
    last_err = None
    for model_name in candidate_models:
        try:
            model = get_chat_model(model_name=model_name, temperature=temperature).bind(
                response_format={"type": "json_object"}
            )
            response = await model.ainvoke(messages)
            content = response.content
            if isinstance(content, str):
                return json.loads(content)
            elif isinstance(content, dict):
                return content
        except Exception as e:
            last_err = e
            logger.warning(f"[LangChain JSON {provider}] Model {model_name} failed: {e}. Trying fallback...")

    logger.error(f"[LangChain JSON {provider}] All models failed. Last error: {last_err}")
    return {}
