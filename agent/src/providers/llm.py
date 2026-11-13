"""Backward-compatible public imports for the in-process provider gateway."""

from src.providers.adapters import ChatOpenAIWithReasoning as ChatOpenAIWithReasoning
from src.providers.adapters import (
    _is_anthropic_temperature_unsupported_error as _is_anthropic_temperature_unsupported_error,
)
from src.providers.adapters import (
    _is_responses_required_error as _is_responses_required_error,
)
from src.providers.adapters import (  # compatibility imports
    _targets_opencode as _targets_opencode,
)
from src.providers.configuration import _ensure_dotenv as _ensure_dotenv
from src.providers.configuration import _sync_provider_env as _sync_provider_env
from src.providers.diagnostics import provider_diagnostics as provider_diagnostics
from src.providers.gateway import build_llm as build_llm
from src.providers.adapters import uses_responses_api as uses_responses_api
