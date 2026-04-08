# CLAUDE.md

## Project Overview

**perteval** — A unified evaluation framework for single-cell perturbation prediction.

- **Repo:** `vc-perturb-eval`, **Package:** `perteval`, **Import:** `import perteval`
- **Python:** 3.11+, **Build:** UV + hatchling
- **Data format:** AnnData (h5ad) core

## Architecture

Four-layer modular design. Each layer depends only on layers below.

```
L4  perteval.bench    — BenchmarkRunner, Evaluator, Compare, TaskManager, EvalResult, CLI
L3  perteval.metrics  — MetricRegistry (lazy), functional API (expression/de/distribution)
L2  perteval.models   — PerturbationModel Protocol, ModelRegistry, MeanControl baseline
L1  perteval.data     — PerturbationData (frozen dataclass), Splitter, LocalAccessor
```

Shared: `perteval._registry.Registry[T]` — generic lazy-loading registry used by L2 and L3.

## Key Files

- `src/perteval/bench/runner.py` — BenchmarkRunner orchestrates: load data → split → train → predict → evaluate
- `src/perteval/bench/evaluator.py` — Evaluator computes metrics per perturbation, returns EvalResult
- `src/perteval/data/types.py` — PerturbationData frozen dataclass (inter-layer contract)
- `src/perteval/metrics/registry.py` — metric_registry with 5 built-in metrics (lazy-loaded)
- `src/perteval/models/baselines/mean_control.py` — MeanControl baseline
- `benchmarks/norman19.yaml` — benchmark YAML definition
- `docs/superpowers/specs/2026-04-07-perteval-design.md` — full design spec
- `docs/superpowers/plans/2026-04-07-perteval-v1.md` — v1 implementation plan

## Common Commands

```bash
uv sync --dev                    # install with dev deps
uv run pytest -v                 # run all tests (54 tests, ~1s)
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run perteval --version        # verify CLI
uv run perteval list metrics     # list available metrics
uv run perteval list models      # list available models
uv run perteval run --benchmark norman19 --model mean_control --data-dir ./data
```

## Conventions

- Commit messages in English
- TDD: write tests first, verify fail, implement, verify pass
- Polars (not pandas) for DataFrames
- Type hints throughout, frozen dataclasses for data contracts
- Private modules prefixed with `_`
- Metrics: pure functions in `metrics/functional/`, registered lazily in `metrics/registry.py`
- Models: implement `PerturbationModel` Protocol (load/train/predict), register in `models/registry.py`
- Benchmarks: YAML files in `benchmarks/`, new fields always optional with defaults

## Current State (v1 complete)

- 54 tests passing, lint clean
- Full pipeline working: data → split → train → predict → evaluate → JSON/CSV export
- CLI: `perteval run`, `perteval evaluate`, `perteval list`
- Built-in: MeanControl baseline, 5 metrics (pearson_delta, mse, mae, overlap_at_k, edistance)
- Notebook tutorial: `notebooks/quickstart.ipynb`

## Next Steps

See GitHub Issues for current tasks. 
