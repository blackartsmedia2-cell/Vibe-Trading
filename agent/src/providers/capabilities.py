"""Provider capability definitions for OpenAI-compatible chat adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from time import monotonic
from typing import Mapping, Optional

from src.providers.registry import capability_records, provider_catalog


@dataclass(frozen=True)
class ProviderCapabilities:
    """Provider-specific payload and diagnostic behavior.

    Args:
        name: Canonical provider name.
        api_key_env: Provider-specific API key environment variable.
        base_url_env: Provider-specific base URL environment variable.
        capture_reasoning: Whether inbound reasoning fields should be preserved.
        send_reasoning_content: Whether outbound assistant history must include
            ``reasoning_content``.
        gemini_thought_signatures: Whether Gemini OpenAI-compatible tool-call
            thought signatures should be round-tripped.
        normalize_assistant_content: Whether assistant ``content=None`` should
            be normalized to ``""`` for strict providers.
        openrouter_reasoning_body: Whether ``extra_body.reasoning`` is a valid
            OpenRouter request option.
        top_level_reasoning_effort: Whether the provider accepts a top-level
            ``reasoning_effort`` field on ``/v1/chat/completions``. **A positive
            allowlist, and it stays one.** Speaking the OpenAI wire format does
            not imply accepting every OpenAI field: an endpoint that validates
            its request body strictly rejects the unknown key and the call
            fails, so a provider gets this only once someone has watched a real
            request to it succeed. Leave it False until then — the cost of False
            is that ``LANGCHAIN_REASONING_EFFORT`` has no effect there, and the
            cost of a wrong True is that every request fails.
        reasoning_effort_models: Optional per-model scoping for
            ``top_level_reasoning_effort``. ``None`` (the default) means every
            model of the provider accepts the field; a tuple means only models
            whose name starts with one of the entries do — prefixes, so
            ``"kimi-k3.1-pro"`` also covers ``kimi-k3.1-pro-highspeed``. An
            unmatched model is a no-op, not an error, mirroring the asymmetry
            of the provider-level flag. Ignored when the flag is False.
        default_headers: Provider-scoped headers passed to ChatOpenAI.
        native_adapter_package: Optional native adapter package to report.
    """

    name: str
    api_key_env: Optional[str]
    base_url_env: str
    capture_reasoning: bool = False
    send_reasoning_content: bool = False
    gemini_thought_signatures: bool = False
    normalize_assistant_content: bool = False
    openrouter_reasoning_body: bool = False
    top_level_reasoning_effort: bool = False
    reasoning_effort_models: tuple[str, ...] | None = None
    default_headers: Mapping[str, str] = field(default_factory=dict)
    native_adapter_package: Optional[str] = None


# Distribution name from pyproject.toml [project].name.
_DISTRIBUTION_NAME = "vibe-trading-ai"


def _package_version() -> str:
    """Return the installed distribution version for User-Agent headers.

    Returns:
        Installed ``vibe-trading-ai`` version string, or ``"dev"`` when the
        package metadata is unavailable (e.g. an uninstalled source checkout).
    """
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "dev"


_VIBE_USER_AGENT = f"Vibe-Trading/{_package_version()}"


_GH_CLI_TOKEN_TTL_SECONDS = 60.0
_gh_cli_token_cache: tuple[float, str] | None = None
_gh_cli_token_lock = Lock()


def _gh_cli_token() -> str:
    """Return an explicit Copilot token; the SDK handles stored credentials."""
    global _gh_cli_token_cache

    with _gh_cli_token_lock:
        now = monotonic()
        if _gh_cli_token_cache is not None and now - _gh_cli_token_cache[0] < _GH_CLI_TOKEN_TTL_SECONDS:
            return _gh_cli_token_cache[1]

        from src.providers.copilot_auth import resolve_copilot_token

        token = resolve_copilot_token()[0]
        _gh_cli_token_cache = (monotonic(), token)
        return token


# Catalog flags preserve the existing protocol contracts: top-level reasoning
# remains opt-in; Moonshot model/endpoint checks stay in the adapter; opaque
# reasoning and thought-signature fields still round-trip only where declared.
def _catalog_capabilities() -> dict[str, ProviderCapabilities]:
    """Build capability records, retaining identity for canonical aliases."""
    records: dict[str, ProviderCapabilities] = {}
    for name, values in capability_records(_VIBE_USER_AGENT).items():
        caps = ProviderCapabilities(**values)
        canonical = records.get(caps.name)
        records[name] = canonical if canonical == caps else caps
    return records


_PROVIDERS = _catalog_capabilities()


def _infer_from_model(model: str) -> str | None:
    lowered = model.strip().lower()
    if not lowered:
        return None
    if lowered.startswith("gemini"):
        return "gemini"
    if lowered.startswith("deepseek"):
        return "deepseek"
    if lowered.startswith("nvidia/"):
        return "nvidia"
    if lowered.startswith("glm"):
        return "zhipu"
    if "kimi" in lowered or "moonshot" in lowered:
        return "moonshot"
    return None


def get_provider_capabilities(
    provider: str | None = None,
    model: str | None = None,
) -> ProviderCapabilities:
    """Return the capability record for a provider/model pair.

    Args:
        provider: Configured provider name.
        model: Configured model name, used for direct test/adapter inference.

    Returns:
        Provider capability definition. Unknown providers fall back to OpenAI.

    Notes:
        Model-name inference (``_infer_from_model``) activates for the default
        ``"openai"`` provider and empty/None providers. Explicit non-OpenAI
        providers (OpenRouter, DeepSeek, etc.) are never inferred —
        the explicit provider choice always wins.
    """
    normalized = (provider or "").strip().lower().replace("_", "-")
    if normalized == "custom":
        return ProviderCapabilities("custom", None, "OPENAI_BASE_URL", capture_reasoning=True)
    if normalized == "minimax" and model == "MiniMax-M3.1-Flash":
        # Flash returns reasoning separately and requires it on subsequent tool turns.
        return ProviderCapabilities(
            "minimax",
            "MINIMAX_API_KEY",
            "MINIMAX_BASE_URL",
            capture_reasoning=True,
            send_reasoning_content=True,
        )
    if normalized and normalized != "openai":
        return _PROVIDERS.get(normalized, _PROVIDERS["openai"])
    inferred = _infer_from_model(model or "")
    if inferred:
        return _PROVIDERS[inferred]
    return _PROVIDERS.get(normalized, _PROVIDERS["openai"])


def provider_env_names(provider: str | None, model: str | None = None) -> tuple[str | None, str]:
    """Return the API-key and base-URL env names for a provider/model pair."""
    caps = get_provider_capabilities(provider, model)
    return caps.api_key_env, caps.base_url_env


@lru_cache(maxsize=1)
def _provider_default_base_urls() -> dict[str, str]:
    """Canonical endpoint defaults from the validated provider catalog."""
    return {row["name"]: row["default_base_url"] for row in provider_catalog() if row["default_base_url"]}


def get_llm_credentials(
    provider: str | None,
    model: str | None,
    *, environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve legacy-compatible credentials without changing process settings.

    Args:
        provider: Configured provider name.
        model: Configured model name, including legacy model-based inference.
        environment: Captured settings, or None for current environment/login.

    Returns:
        Provider, API key, endpoint and model. Custom profiles read the active
        credential vault; other providers retain the generic OPENAI_* fallback.
    """
    env = os.environ if environment is None else environment
    normalized_provider = (provider or "").strip().lower().replace("_", "-")
    if normalized_provider == "custom":
        from src.providers.custom_profiles import active_provider_credentials

        active = active_provider_credentials()
        return {"provider": "custom", **active}

    caps = get_provider_capabilities(provider, model)
    key_env, base_env = caps.api_key_env, caps.base_url_env

    api_key = (env.get(key_env, "") if key_env else "") or env.get("OPENAI_API_KEY", "")
    if key_env is None:
        api_key = api_key or "ollama"

    if caps.name == "copilot":
        from src.providers.copilot_auth import TOKEN_ENV_NAMES, is_supported_token
        api_key = _gh_cli_token() if environment is None else next(
            (env[name].strip() for name in TOKEN_ENV_NAMES if is_supported_token(env.get(name, "").strip())),
            "",
        )

    base_url = (
        env.get(base_env, "")
        or env.get("OPENAI_BASE_URL", "")
        or env.get("OPENAI_API_BASE", "")
        or _provider_default_base_urls().get(caps.name, "")
    )
    if caps.name == "ollama":
        base_url = base_url.strip().rstrip("/")
        if base_url and not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

    return {
        "provider": (provider or "").strip().lower(),
        "api_key": api_key,
        "base_url": base_url,
        "model": model or "",
    }
