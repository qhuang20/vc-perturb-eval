from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    dataset: str
    metrics: list[str]
    split_method: str = "random"
    split_frac: tuple[float, ...] = (0.8, 0.1, 0.1)
    split_seed: int = 42
    holdout_key: str | None = None
    aggregation: str = "average"


class TaskManager:
    def __init__(self, benchmarks_dir: str) -> None:
        self._dir = Path(benchmarks_dir)

    def list_available(self) -> list[str]:
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.yaml"))

    def get(self, name: str) -> BenchmarkConfig:
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise KeyError(
                f"Benchmark '{name}' not found at {path}. Available: {self.list_available()}"
            )
        with open(path) as f:
            raw = yaml.safe_load(f)
        split_config = raw.get("split", {})
        return BenchmarkConfig(
            name=name,
            dataset=raw["dataset"],
            metrics=raw["metrics"],
            split_method=split_config.get("method", "random"),
            split_frac=tuple(split_config.get("frac", [0.8, 0.1, 0.1])),
            split_seed=split_config.get("seed", 42),
            holdout_key=split_config.get("holdout_key"),
            aggregation=raw.get("aggregation", "average"),
        )
