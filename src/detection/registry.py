"""DetectorRegistry — maps backend name strings to backend classes (NFR-012)."""

from __future__ import annotations

import logging
from typing import Any, Type

from src.detection.base import DetectorBackend

logger = logging.getLogger(__name__)

_registry: dict[str, Type[DetectorBackend]] = {}


def register(name: str, backend_cls: Type[DetectorBackend]) -> None:
    """Register a detection backend under a name.

    Args:
        name: Registry key used in config (e.g. 'yolov12').
        backend_cls: Class implementing DetectorBackend.

    Raises:
        TypeError: When backend_cls does not implement DetectorBackend.
    """
    if not (isinstance(backend_cls, type) and issubclass(backend_cls, object)):
        raise TypeError(f"{backend_cls} must be a class")
    _registry[name] = backend_cls
    logger.debug("Registered detection backend: '%s' -> %s", name, backend_cls.__name__)


def get(name: str) -> Type[DetectorBackend]:
    """Retrieve a registered backend class by name.

    Args:
        name: Registry key.

    Returns:
        The backend class.

    Raises:
        KeyError: When no backend is registered under the given name.
    """
    if name not in _registry:
        available = list(_registry.keys())
        raise KeyError(
            f"No detection backend registered as '{name}'. "
            f"Available: {available}"
        )
    return _registry[name]


def list_backends() -> list[str]:
    """Return all registered backend names."""
    return list(_registry.keys())


class DetectorRegistry:
    """Namespace wrapper exposing the module-level registry as a class (IC-015)."""

    @staticmethod
    def register(name: str, backend_cls: Type[Any]) -> None:
        """Register a backend. See module-level register()."""
        register(name, backend_cls)

    @staticmethod
    def get(name: str) -> Type[DetectorBackend]:
        """Get a backend class by name. See module-level get()."""
        return get(name)

    @staticmethod
    def list_backends() -> list[str]:
        """List all registered backend names."""
        return list_backends()
