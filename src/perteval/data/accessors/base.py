from __future__ import annotations

from typing import Protocol

from anndata import AnnData


class DataAccessor(Protocol):
    def load(self, name: str, **kwargs) -> AnnData: ...
    def list_datasets(self) -> list[str]: ...
