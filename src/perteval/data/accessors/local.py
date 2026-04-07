from __future__ import annotations

from pathlib import Path

import anndata as ad


class LocalAccessor:
    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)

    def load(self, name: str, **kwargs) -> ad.AnnData:
        path = self._data_dir / f"{name}.h5ad"
        if not path.exists():
            available = self.list_datasets()
            raise FileNotFoundError(f"Dataset '{name}' not found at {path}. Available: {available}")
        return ad.read_h5ad(path, **kwargs)

    def list_datasets(self) -> list[str]:
        return sorted(p.stem for p in self._data_dir.glob("*.h5ad"))
