"""Compatibility imports; provider facts live in registry, credentials in settings."""

from src.providers.registry import (
    ProviderCapabilities as ProviderCapabilities,
    get_provider_capabilities as get_provider_capabilities,
    provider_env_names as provider_env_names,
)
from src.providers.settings import get_llm_credentials as get_llm_credentials
