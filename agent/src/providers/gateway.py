"""In-process LLM gateway: catalog-backed routing and adapter construction.

ChatLLM remains the normalized chat/stream interface. This module owns the
selection boundary and does not introduce a server, a second message format,
or automatic cross-provider retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.providers.capabilities import ProviderCapabilities, get_provider_capabilities
from src.providers.registry import provider_backend, provider_catalog


@dataclass(frozen=True)
class ProviderRoute:
    """Credential-free resolved route; safe to inspect before constructing SDKs."""

    provider: str
    model: str
    backend: str
    capabilities: ProviderCapabilities


def resolve_route(provider: str, model: str) -> ProviderRoute:
    """Resolve adapter selection while preserving legacy capability inference.

    Args:
        provider: Explicit provider; aliases use the same backend.
        model: Requested model, never rewritten based on its name.

    Returns:
        Route separating the selected backend from inferred wire capabilities.

    Raises:
        RuntimeError: If the model is empty.
    """
    if not model:
        raise RuntimeError("LANGCHAIN_MODEL_NAME is not set")
    return ProviderRoute(
        provider=provider.lower(),
        model=model,
        backend=provider_backend(provider),
        capabilities=get_provider_capabilities(provider, model),
    )


def build_model(*, model_name: str | None = None, callbacks: Any = None) -> Any:
    """Create a fresh adapter through the common route and compatibility factory.

    Args:
        model_name: Optional override for the configured model.
        callbacks: Optional LangChain callbacks.

    Returns:
        Provider adapter consumed by ChatLLM and other existing LLM callers.
    """
    from src.providers import llm as compat
    from src.providers.factory import build_adapter

    compat._sync_provider_env()
    cfg = compat.get_env_config().llm
    route = resolve_route(cfg.langchain_provider, model_name or cfg.langchain_model_name.strip())
    return build_adapter(route, model_name=model_name, callbacks=callbacks)


def catalog() -> list[dict[str, Any]]:
    """Return public provider metadata without loading SDKs or credentials."""
    return provider_catalog()
