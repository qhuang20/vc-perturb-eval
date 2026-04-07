from __future__ import annotations

from typing import Protocol

from anndata import AnnData


class PerturbationModel(Protocol):
    name: str

    def load(self, path: str | None = None, **kwargs) -> None: ...

    def train(self, adata: AnnData, **kwargs) -> None: ...

    def predict(self, control_adata: AnnData, perturbations: list[str], **kwargs) -> AnnData: ...
