"""Base types for the perteval metric system."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class MetricType(Enum):
    EXPRESSION = "expression"
    DE = "de"
    DISTRIBUTION = "distribution"


class BestValue(Enum):
    ZERO = "zero"
    ONE = "one"
    NONE = "none"


@dataclass
class MetricInfo:
    name: str
    func: Callable[..., float] | str
    metric_type: MetricType
    best_value: BestValue
    description: str
