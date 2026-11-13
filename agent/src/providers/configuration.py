"""Immutable per-client configuration; environment compatibility at one boundary."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Mapping

from pydantic import ConfigDict, Field, model_validator

from src.config.accessor import reset_env_config
from src.config.env_schema import LLMConfig, _numeric_type
from src.providers.capabilities import get_llm_credentials
from src.providers.registry import provider_backend

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)
CONFIG_LOCK = RLock()
AGENT_DIR = Path(__file__).resolve().parents[2]
_ENV_CANDIDATES = [Path.home() / ".vibe-trading" / ".env", AGENT_DIR / ".env", Path.cwd() / ".env"]
_ENV_LABELS = ("~/.vibe-trading/.env", "<AGENT_DIR>/.env", "<CWD>/.env")
_dotenv_loaded = False


class ModelOptions(LLMConfig):
    """Reuse the existing schema and defaults without reading ambient state."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)
    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY", default="", repr=False)

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, data):
        values = dict(data)
        for name, info in cls.model_fields.items():
            key = info.alias if info.alias in values else name
            numeric = _numeric_type(info.annotation)
            if key in values and numeric and isinstance(values[key], str):
                try:
                    values[key] = numeric(values[key])
                except ValueError:
                    values.pop(key)
        return values


@dataclass(frozen=True)
class ProviderConfig:
    """One resolved connection and generation snapshot; secrets never enter repr."""

    provider: str
    model: str
    backend: str
    base_url: str = field(repr=False)
    options: ModelOptions = field(repr=False)
    api_key: str = field(repr=False, default="")
    auth_token: str = field(repr=False, default="")
    environment: Mapping[str, str] = field(repr=False, default_factory=dict)


def resolve_config(*, model_name: str | None = None, environment: Mapping[str, str] | None = None) -> ProviderConfig:
    """Resolve once from supplied settings and, for custom providers, the vault.

    Args:
        model_name: Optional model override.
        environment: Complete connection settings, or None for current settings.

    Returns:
        An immutable config for one client and all of its subsequent requests.
    """
    with CONFIG_LOCK:
        if environment is None:
            _ensure_dotenv()
        values = dict(os.environ if environment is None else environment)
        options = ModelOptions(**values)
        provider = options.langchain_provider.strip().lower()
        model = model_name or options.langchain_model_name.strip()
        creds = get_llm_credentials(provider, model, environment=values)
        if provider == "custom":
            model = model_name or creds["model"]
        backend = provider_backend(provider)
        if backend == "anthropic":
            creds = {
                "api_key": values.get("ANTHROPIC_API_KEY", ""),
                "base_url": values.get("ANTHROPIC_BASE_URL")
                or values.get("ANTHROPIC_API_URL")
                or "https://api.anthropic.com",
            }
        if backend in {"openai_codex", "claude_cli"}:
            creds["api_key"] = ""
        if backend == "openai_codex":
            creds["base_url"] = options.openai_codex_base_url
        return ProviderConfig(
            provider,
            model,
            backend,
            creds["base_url"],
            options,
            creds["api_key"],
            values.get("ANTHROPIC_AUTH_TOKEN", ""),
            MappingProxyType(values),
        )


def _redact_env_source(loaded: Path | None) -> str:
    """Map a resolved `.env` candidate to a stable, leak-free label.

    Returns a symbolic slot label (never the absolute path) so a stale
    or shadowed `.env` stays diagnosable without exposing the OS
    username, home, or CWD (CWE-209). A candidate outside the fixed
    list (e.g. one injected by a test) collapses to a generic
    placeholder rather than echoing a real path.
    """
    if loaded is None:
        return "none (no .env file found)"
    for label, candidate in zip(_ENV_LABELS, _ENV_CANDIDATES):
        if loaded == candidate:
            return label
    return "<.env>"


def _load_env_file(path: Path) -> None:
    """Load a single .env file into os.environ without clobbering real vars.

    Exported environment variables are explicit operator intent and outrank
    the file; ``.env`` only fills the gaps (pinned by
    ``test_dispatch_connector_never_overrides_a_real_environment_variable``).
    """
    if load_dotenv is not None:
        load_dotenv(dotenv_path=path, override=False)
    else:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")


def _ensure_dotenv() -> None:
    """Load `.env` from the first found candidate path."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    loaded = None
    for candidate in _ENV_CANDIDATES:
        if candidate.exists():
            _load_env_file(candidate)
            loaded = candidate
            break
    if loaded is not None:
        reset_env_config()
    _dotenv_loaded = True
    logger.info("dotenv resolved from %s", _redact_env_source(loaded))


def _sync_provider_env() -> None:
    """Legacy initialization hook; no longer rewrites another provider's keys."""
    with CONFIG_LOCK:
        _ensure_dotenv()
        reset_env_config()
