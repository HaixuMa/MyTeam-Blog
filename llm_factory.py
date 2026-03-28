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

        mk = dict(cfg.model_extra_body or {})
        mk.pop("enable_thinking", None)
        mk.pop("thinking_budget", None)
        llm = ChatOpenAI(
            model=cfg.model_name,
            temperature=cfg.model_temperature,
            timeout=cfg.request_timeout_s,
            model_kwargs=mk,
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

        base_kwargs = {
            "model": cfg.model_name,
            "temperature": cfg.model_temperature,
            "timeout": cfg.request_timeout_s,
            "base_url": cfg.model_base_url,
            "streaming": True,
        }
        eb = dict(cfg.model_extra_body or {})
        eb.pop("enable_thinking", None)
        eb.pop("thinking_budget", None)
        extra_body = eb or None
        if extra_body:
            try:
                llm = ChatOpenAI(**base_kwargs, extra_body=extra_body)
            except TypeError:
                llm = ChatOpenAI(**base_kwargs, model_kwargs={"extra_body": extra_body})
        else:
            llm = ChatOpenAI(**base_kwargs)
        object.__setattr__(
            llm,
            "_raw_openai_compatible",
            {
            "base_url": cfg.model_base_url,
            "api_key": cfg.openai_api_key,
            "model": cfg.model_name,
            "temperature": cfg.model_temperature,
            "timeout_s": cfg.request_timeout_s,
            },
        )
        return llm, ModelInfo(provider=provider, model_name=cfg.model_name)

    raise ConfigError(
        "Unsupported MODEL_PROVIDER. Use one of: openai, anthropic, openai_compatible"
    )
