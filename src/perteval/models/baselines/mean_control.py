from __future__ import annotations

import anndata as ad
import numpy as np
from anndata import AnnData


class MeanControl:
    name: str = "mean_control"

    def __init__(self) -> None:
        self._control_mean: np.ndarray | None = None
        self._gene_names: list[str] = []

    def load(self, path: str | None = None, **kwargs) -> None:
        pass

    def train(
        self,
        adata: AnnData,
        perturbation_key: str = "perturbation",
        control_key: str = "control",
        **kwargs,
    ) -> None:
        control_mask = adata.obs[perturbation_key] == control_key
        control_cells = np.asarray(adata[control_mask].X)
        self._control_mean = control_cells.mean(axis=0)
        self._gene_names = list(adata.var_names)

    def predict(self, control_adata: AnnData, perturbations: list[str], **kwargs) -> AnnData:
        if self._control_mean is None:
            raise RuntimeError("Model not trained. Call train() first.")
        n_perts = len(perturbations)
        X = np.tile(self._control_mean, (n_perts, 1)).astype(np.float32)  # noqa: N806
        predicted = ad.AnnData(X=X, obs={"perturbation": perturbations})
        predicted.var_names = self._gene_names
        predicted.obs_names = [f"pred_{p}" for p in perturbations]
        return predicted
