from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from config import AppConfig


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model_name: str


class ConfigError(RuntimeError):
    pass


def create_chat_model(cfg: AppConfig) -> tuple[BaseChatModel, ModelInfo]:
    provider = cfg.model_provider

    if provider == "openai":
        if not cfg.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required for MODEL_PROVIDER=openai")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=cfg.model_name,
            temperature=cfg.model_temperature,
            timeout=cfg.request_timeout_s,
        )
        return llm, ModelInfo(provider=provider, model_name=cfg.model_name)

    if provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is required for MODEL_PROVIDER=anthropic")
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=cfg.model_name,
            temperature=cfg.model_temperature,
            timeout=cfg.request_timeout_s,
        )
        return llm, ModelInfo(provider=provider, model_name=cfg.model_name)

    if provider == "openai_compatible":
        if not cfg.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required for MODEL_PROVIDER=openai_compatible")
        if not cfg.model_base_url:
            raise ConfigError("MODEL_BASE_URL is required for MODEL_PROVIDER=openai_compatible")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=cfg.model_name,
            temperature=cfg.model_temperature,
            timeout=cfg.request_timeout_s,
            base_url=cfg.model_base_url,
        )
        return llm, ModelInfo(provider=provider, model_name=cfg.model_name)

    raise ConfigError(
        "Unsupported MODEL_PROVIDER. Use one of: openai, anthropic, openai_compatible"
    )

