from __future__ import annotations

from perteval._registry import Registry

model_registry: Registry = Registry("model")
model_registry.register("mean_control", "perteval.models.baselines.mean_control:MeanControl")
