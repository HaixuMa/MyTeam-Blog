from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    environment: str
    log_level: str
    data_dir: Path
    checkpoint_sqlite_path: Path

    model_provider: str
    model_name: str
    model_temperature: float
    model_base_url: str | None
    request_timeout_s: float

    tavily_api_key: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None

    max_agent_retries: int
    max_image_retries: int
    max_audit_rounds: int

    tool_rate_limit_per_minute: int


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def load_config(project_root: Path) -> AppConfig:
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", str(project_root / "data"))).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(
        os.getenv("CHECKPOINT_SQLITE_PATH", str(data_dir / "checkpoints.sqlite3"))
    ).resolve()

    return AppConfig(
        environment=os.getenv("APP_ENV", "local"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        data_dir=data_dir,
        checkpoint_sqlite_path=checkpoint_path,
        model_provider=os.getenv("MODEL_PROVIDER", "openai").strip().lower(),
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_temperature=_env_float("MODEL_TEMPERATURE", 0.2),
        model_base_url=os.getenv("MODEL_BASE_URL") or None,
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 60.0),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        max_agent_retries=_env_int("MAX_AGENT_RETRIES", 3),
        max_image_retries=_env_int("MAX_IMAGE_RETRIES", 2),
        max_audit_rounds=_env_int("MAX_AUDIT_ROUNDS", 2),
        tool_rate_limit_per_minute=_env_int("TOOL_RATE_LIMIT_PER_MINUTE", 20),
    )

