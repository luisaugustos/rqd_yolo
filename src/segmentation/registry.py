"""SegmentorRegistry — maps backend name strings to backend classes (NFR-012)."""

from __future__ import annotations

import logging
from typing import Any, Type

logger = logging.getLogger(__name__)

_registry: dict[str, Type[Any]] = {}


def register(name: str, backend_cls: Type[Any]) -> None:
    """Register a segmentation backend.

    Args:
        name: Registry key (e.g. 'sam2', 'none').
        backend_cls: Class implementing SegmentorBackend.
    """
    _registry[name] = backend_cls
    logger.debug("Registered segmentation backend: '%s' -> %s", name, backend_cls.__name__)


def get(name: str) -> Type[Any]:
    """Retrieve a registered backend class.

    Args:
        name: Registry key.

    Returns:
        Backend class.

    Raises:
        KeyError: When no backend is registered under the name.
    """
    if name not in _registry:
        available = list(_registry.keys())
        raise KeyError(
            f"No segmentation backend registered as '{name}'. Available: {available}"
        )
    return _registry[name]


def list_backends() -> list[str]:
    """List all registered segmentation backend names."""
    return list(_registry.keys())


class SegmentorRegistry:
    """Namespace wrapper exposing the module-level registry as a class."""

    @staticmethod
    def register(name: str, backend_cls: Type[Any]) -> None:
        """Register a backend. See module-level register()."""
        register(name, backend_cls)

    @staticmethod
    def get(name: str) -> Type[Any]:
        """Get a backend class by name."""
        return get(name)
