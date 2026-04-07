"""Base types for the perteval metric system."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


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
