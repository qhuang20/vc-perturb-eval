"""Generic lazy-loading registry shared across all perteval components."""

from __future__ import annotations

import importlib
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named registry that supports both concrete instances and lazy string references.

    Lazy references use the format "module.path:attribute" and are resolved
    on first access via get().
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T | str] = {}

    def register(self, name: str, entry: T | str) -> None:
        """Register an entry. `entry` can be a concrete instance or a lazy
        string like 'perteval.metrics.functional.expression:pearson_delta'."""
        if name in self._entries:
            raise ValueError(f"'{name}' is already registered in {self._name} registry")
        self._entries[name] = entry

    def get(self, name: str) -> T:
        """Retrieve an entry by name, resolving lazy strings on first access."""
        if name not in self._entries:
            raise KeyError(f"'{name}' not found in {self._name} registry")
        entry = self._entries[name]
        if isinstance(entry, str) and ":" in entry:
            resolved = self._resolve(entry)
            self._entries[name] = resolved
            return resolved
        return entry

    def list_available(self) -> list[str]:
        """Return sorted list of all registered names."""
        return sorted(self._entries.keys())

    @staticmethod
    def _resolve(entry_string: str) -> T:
        """Resolve 'module.path:attribute' to the actual object."""
        module_path, _, attr_name = entry_string.partition(":")
        if not attr_name:
            raise ValueError(f"Lazy entry must be 'module:attribute' format, got '{entry_string}'")
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
