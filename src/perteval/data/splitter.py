from __future__ import annotations

import numpy as np
from anndata import AnnData


class Splitter:
    @staticmethod
    def split(
        adata: AnnData,
        method: str = "random",
        frac: tuple[float, ...] = (0.8, 0.1, 0.1),
        holdout_key: str | None = None,
        seed: int = 42,
    ) -> dict[str, AnnData]:
        match method:
            case "random":
                return Splitter._random_split(adata, frac, seed)
            case "transfer":
                if holdout_key is None:
                    raise ValueError("transfer split requires holdout_key")
                return Splitter._transfer_split(adata, frac, holdout_key, seed)
            case _:
                raise ValueError(f"Unknown split method '{method}'. Choose from: random, transfer")

    @staticmethod
    def _random_split(adata: AnnData, frac: tuple[float, ...], seed: int) -> dict[str, AnnData]:
        rng = np.random.default_rng(seed)
        n = adata.n_obs
        indices = rng.permutation(n)
        train_end = int(n * frac[0])
        val_end = train_end + int(n * frac[1])
        return {
            "train": adata[indices[:train_end]].copy(),
            "val": adata[indices[train_end:val_end]].copy(),
            "test": adata[indices[val_end:]].copy(),
        }

    @staticmethod
    def _transfer_split(
        adata: AnnData, frac: tuple[float, ...], holdout_key: str, seed: int
    ) -> dict[str, AnnData]:
        rng = np.random.default_rng(seed)
        all_values = adata.obs[holdout_key].unique().tolist()
        holdout_values = [v for v in all_values if v != "control"]
        rng.shuffle(holdout_values)
        n = len(holdout_values)
        train_end = int(n * frac[0])
        val_end = train_end + int(n * frac[1])
        train_values = set(holdout_values[:train_end]) | {"control"}
        val_values = set(holdout_values[train_end:val_end]) | {"control"}
        test_values = set(holdout_values[val_end:]) | {"control"}
        return {
            "train": adata[adata.obs[holdout_key].isin(train_values)].copy(),
            "val": adata[adata.obs[holdout_key].isin(val_values)].copy(),
            "test": adata[adata.obs[holdout_key].isin(test_values)].copy(),
        }
