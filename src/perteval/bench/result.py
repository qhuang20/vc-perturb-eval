"""Self-describing evaluation result container."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import polars as pl


@dataclass
class EvalResult:
    per_perturbation: pl.DataFrame
    aggregated: pl.DataFrame
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "timestamp" not in self.config:
            self.config["timestamp"] = datetime.now(UTC).isoformat()

    def to_json(self, path: str) -> None:
        data = {
            "config": self.config,
            "per_perturbation": self.per_perturbation.to_dicts(),
            "aggregated": self.aggregated.to_dicts(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def to_csv(self, path: str) -> None:
        self.per_perturbation.write_csv(f"{path}_per_perturbation.csv")
        self.aggregated.write_csv(f"{path}_aggregated.csv")
