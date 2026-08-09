"""Expose the intentional public API for validated runtime configuration."""

from config.runtime import (
    ALLOWED_SPLITS,
    RuntimeConfig,
    apply_cli_overrides,
    load_runtime_config,
)

__all__ = [
    "ALLOWED_SPLITS",
    "RuntimeConfig",
    "apply_cli_overrides",
    "load_runtime_config",
]
