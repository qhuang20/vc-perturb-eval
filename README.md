# perteval

[![CI](https://github.com/qhuang20/vc-perturb-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/qhuang20/vc-perturb-eval/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified evaluation framework for single-cell perturbation prediction.

## Why perteval?

Evaluating perturbation prediction models today means stitching together ad-hoc scripts, inconsistent metrics, and non-reproducible splits. Existing tools either cover only evaluation ([cell-eval](https://github.com/arc-institute/cell-eval)), only training ([PerturBench](https://github.com/altoslabs/perturbench)), or lack modularity ([scPerturBench](https://github.com/bm2-lab/scPerturBench)).

**perteval** provides the full pipeline — data loading, model training, metric computation, and benchmark orchestration — in a modular, layered design where each component works independently or together.

## Installation

```bash
pip install perteval          # core (metrics only, lightweight)
pip install perteval[models]  # + PyTorch model support
pip install perteval[all]     # everything
```

**From source (development):**

```bash
git clone https://github.com/qhuang20/vc-perturb-eval.git
cd vc-perturb-eval
uv sync --dev
```

## Quick Start

### Evaluate predictions directly

```python
import anndata as ad
from perteval import PerturbationData, Evaluator

predicted = ad.read_h5ad("predicted.h5ad")
ground_truth = ad.read_h5ad("ground_truth.h5ad")

data = PerturbationData(predicted=predicted, ground_truth=ground_truth)
result = Evaluator().evaluate(data, metrics=["pearson_delta", "mse", "edistance"])

print(result.per_perturbation)  # per-perturbation scores
print(result.aggregated)        # mean/std/min/max

result.to_json("results.json")  # reproducible output with config metadata
result.to_csv("results")        # human-friendly CSV export
```

### Run a full benchmark

```python
from perteval.bench import BenchmarkRunner, Compare

runner = BenchmarkRunner(
    benchmarks=["norman19"],
    models=["mean_control"],
    benchmarks_dir="benchmarks",
    data_dir="data",
)
results = runner.run()

Compare.from_results(results).summary()
```

### Use metrics standalone

```python
from perteval.metrics.functional.expression import pearson_delta, mse
from perteval.metrics.functional.distribution import edistance

pearson_delta(pred_mean, truth_mean)  # → float
edistance(pred_cells, truth_cells)    # → float
```

### CLI

```bash
perteval evaluate --predicted pred.h5ad --ground-truth real.h5ad --metrics pearson_delta mse
perteval run --benchmark norman19 --model mean_control --data-dir ./data
perteval list metrics
```

## Architecture

Four independent layers, each usable on its own:

```
┌───────────────────────────────────────────────┐
│              perteval.bench (L4)              │
│  BenchmarkRunner · Compare · TaskManager      │
│  YAML benchmark definitions · CLI             │
├───────────────────────────────────────────────┤
│             perteval.metrics (L3)             │
│  MetricRegistry (lazy) · Functional API       │
│  pearson_delta · mse · mae · overlap_at_k     │
│  edistance                                    │
├───────────────────────────────────────────────┤
│             perteval.models (L2)              │
│  PerturbationModel Protocol · ModelRegistry   │
│  Built-in: MeanControl baseline               │
├───────────────────────────────────────────────┤
│              perteval.data (L1)               │
│  PerturbationData · Splitter · LocalAccessor  │
└───────────────────────────────────────────────┘
```

- **L1 Data** — Load datasets, split (random/transfer), validate predicted vs ground-truth pairs
- **L2 Models** — Protocol-based model interface (`load`/`train`/`predict`), lazy registry
- **L3 Metrics** — Pure functional compute kernels + lazy metric registry with plugin support
- **L4 Bench** — Orchestrate multi-model × multi-dataset evaluations from YAML configs

Each layer depends only on layers below. `pip install perteval` gives you L3 (metrics) with minimal dependencies. Add `[models]` or `[all]` for the full stack.

## Defining Benchmarks

Benchmarks are YAML files — no code changes needed to add new ones:

```yaml
# benchmarks/norman19.yaml
dataset: norman19
metrics: [pearson_delta, mse, mae, overlap_at_k, edistance]
split:
  method: transfer
  holdout_key: perturbation
  frac: [0.64, 0.16, 0.2]
  seed: 42
aggregation: average
```

## Adding Your Own Model

Implement the `PerturbationModel` protocol:

```python
class MyModel:
    name = "my_model"

    def load(self, path=None, **kwargs):
        ...

    def train(self, adata, **kwargs):
        ...

    def predict(self, control_adata, perturbations, **kwargs):
        # Return AnnData with predicted expression
        ...
```

Register it:

```python
from perteval.models import model_registry
model_registry.register("my_model", MyModel)
```

## Available Metrics

| Metric | Type | Best | Description |
|--------|------|------|-------------|
| `pearson_delta` | expression | 1 | Pearson correlation of mean expression shift |
| `mse` | expression | 0 | Mean squared error |
| `mae` | expression | 0 | Mean absolute error |
| `overlap_at_k` | DE | 1 | Top-k DE gene overlap |
| `edistance` | distribution | 0 | Energy distance between cell populations |

## Development

```bash
uv sync --dev           # install with dev dependencies
uv run pytest -v        # run tests (54 tests)
uv run ruff check src/  # lint
uv run ruff format src/ # format
```

## License

MIT
