# perteval — Design Spec

> A unified evaluation framework for single-cell perturbation prediction
> Date: 2026-04-07

---

## 1. Overview

**perteval** is a modular, layered Python framework for evaluating single-cell perturbation prediction models. It covers the full pipeline — data loading, model training/inference, metric computation, and benchmark orchestration — while allowing users to use any layer independently.

**Naming:** repo `vc-perturb-eval`, package `perteval`, `import perteval`

**Target users:** Initially the team's own evaluation workflow, designed from day one for open-source release (docs, API, PyPI publishing).

**Python:** 3.11+  
**Build:** UV + hatchling  
**Data format:** AnnData core, MuData reserved for future multimodal extension

---

## 2. Architecture

Four independent layers, each depending only on layers below:

```
┌─────────────────────────────────────────────────────┐
│                   perteval.bench (L4)                │
│   BenchmarkRunner: multi-model × multi-dataset      │
│   Compare: result comparison, evaluate_many          │
│   YAML benchmark definitions + TaskManager           │
├─────────────────────────────────────────────────────┤
│                  perteval.metrics (L3)               │
│   MetricRegistry: lazy loading + entry_points        │
│   Functional API: pure compute functions             │
│   OOP API: update/compute streaming + Collection     │
│   Wrappers: Bootstrap, CelltypeWise, DEGMask         │
├─────────────────────────────────────────────────────┤
│                  perteval.models (L2)                │
│   PerturbationModel Protocol: load/train/predict     │
│   Built-in: CPA, GEARS, MeanControl                 │
│   Registry: lazy loading, user-registerable           │
├─────────────────────────────────────────────────────┤
│                   perteval.data (L1)                 │
│   DataAccessor Protocol: HuggingFace/scperturb/local │
│   Splitter: random, transfer, combination            │
│   Types: PerturbationData (frozen dataclass)         │
└─────────────────────────────────────────────────────┘
```

**Core principles:**
- Each layer depends only on layers below, never upward
- L3 (metrics) is the minimum installable unit — `pip install perteval`
- L1-L2 installed via optional dependencies
- L4 orchestrates the other three layers; it is the primary user entry point

**Data flow:**
```
DataAccessor.load("norman19") → AnnData
  → Splitter.split(adata, method="transfer") → train/val/test AnnData
    → model.train(train_adata)
    → model.predict(control_adata, perturbations) → predicted AnnData
      → PerturbationData(predicted, ground_truth)  # frozen, validated
        → Evaluator.evaluate(data, metrics=[...], aggregation="logfc")
          → EvalResult {JSON + CSV}
```

---

## 3. L1 — perteval.data

### DataAccessor Protocol

```python
class DataAccessor(Protocol):
    def load(self, name: str, **kwargs) -> AnnData: ...
    def list_datasets(self) -> list[str]: ...
```

**Built-in accessors:**
- `HuggingFaceAccessor` — downloads from HF Datasets, caching via HF's `~/.cache/huggingface/`
- `ScPerturbAccessor` — downloads from scperturb standardized datasets
- `LocalAccessor` — reads local h5ad paths

### Splitter

```python
class Splitter:
    @staticmethod
    def split(
        adata: AnnData,
        method: str = "random",        # random | transfer | combination
        frac: tuple = (0.8, 0.1, 0.1),
        holdout_key: str | None = None, # for transfer: which key to hold out
        seed: int = 42,
    ) -> dict[str, AnnData]:           # {"train": ..., "val": ..., "test": ...}
```

### PerturbationData (inter-layer contract)

```python
@dataclass(frozen=True)
class PerturbationData:
    predicted: AnnData
    ground_truth: AnnData
    perturbation_key: str = "perturbation"
    control_key: str = "control"
    gene_names: np.ndarray  # validated: gene order matches between predicted and ground_truth
```

Construction validates: gene alignment, perturbation label match, shape consistency. Fails loudly on mismatch — no silent correction.

### First-version datasets (3-5)

- Norman et al. 2019
- Replogle et al. 2022
- 2-3 additional classic Perturb-seq datasets

---

## 4. L2 — perteval.models

### PerturbationModel Protocol

```python
class PerturbationModel(Protocol):
    name: str

    def load(self, path: str | None = None, **kwargs) -> None:
        """Load pretrained weights or initialize model."""
        ...

    def train(self, adata: AnnData, **kwargs) -> None:
        """Train the model. Internal logic is the model's responsibility."""
        ...

    def predict(
        self,
        control_adata: AnnData,
        perturbations: list[str],
        **kwargs,
    ) -> AnnData:
        """Given control cells and perturbation list, return predicted AnnData."""
        ...
```

**Design decisions:**
- `train()` returns nothing — training state lives inside the model, framework doesn't manage it
- `predict()` returns AnnData — consistent with L1 data format, feeds directly into `PerturbationData`
- `load()` supports loading from local path or HuggingFace pretrained weights
- All methods accept `**kwargs` for model-specific parameters
- Training is delegated (Branch 7): the framework calls `model.train(adata)` but never controls the training loop. A convenience trainer may be added later as optional, never replacing delegation.

### ModelRegistry

```python
model_registry = Registry[PerturbationModel]()

# Built-in models (lazy)
model_registry.register("mean_control", "perteval.models.baselines:MeanControl")
model_registry.register("cpa", "perteval.models.cpa:CPAWrapper")
model_registry.register("gears", "perteval.models.gears:GEARSWrapper")

# User-defined
model_registry.register("my_model", my_model_instance)
```

### First-version built-in models (3)

- `MeanControl` — baseline, prediction = control cell mean, zero training
- `CPAWrapper` — wraps CPA, delegates to its internal Lightning training
- `GEARSWrapper` — wraps GEARS, delegates to its internal training loop

Wrappers only adapt the interface; they do not alter the model's internal training logic.

---

## 5. L3 — perteval.metrics

### MetricRegistry (lazy loading + entry_points)

```python
metric_registry = Registry[MetricInfo]()

metric_registry.register(
    name="pearson_delta",
    entry="perteval.metrics.functional.expression:pearson_delta",  # lazy
    metric_type=MetricType.EXPRESSION,
    best_value=BestValue.ONE,
    description="Pearson correlation of mean expression shift",
)
```

Built-in metrics registered lazily at module level. Third-party metrics discoverable via Python `entry_points`:

```toml
# In a third-party package's pyproject.toml
[project.entry-points."perteval.metrics"]
my_metric = "my_package.metrics:MyMetric"
```

### MetricType

```python
class MetricType(Enum):
    EXPRESSION = "expression"      # based on expression matrix (pearson, mse, mae, cosine)
    DE = "de"                      # based on differential expression (overlap@k, spearman, direction_match)
    DISTRIBUTION = "distribution"  # based on distribution (edistance, mmd)
```

### Functional API (pure functions)

```python
# perteval.metrics.functional
def pearson_delta(pred: np.ndarray, truth: np.ndarray) -> float: ...
def mse(pred: np.ndarray, truth: np.ndarray) -> float: ...
def overlap_at_k(pred_de: pd.DataFrame, truth_de: pd.DataFrame, k: int = 20) -> float: ...
def edistance(pred_cells: np.ndarray, truth_cells: np.ndarray) -> float: ...
```

Pure numpy/scipy compute kernels. No AnnData dependency. Can be used standalone.

### OOP API (streaming + stateful)

```python
class PearsonDelta(Metric):
    def __init__(self):
        self.add_state("pred_sums", default=[])
        self.add_state("truth_sums", default=[])

    def update(self, data: PerturbationData, perturbation: str) -> None:
        # Accumulate statistics for a single perturbation
        ...

    def compute(self) -> dict[str, float]:
        # Return per-perturbation results
        ...
```

For large datasets: batch `update()`, then `compute()` once.

### MetricCollection

```python
collection = MetricCollection([
    "pearson_delta", "mse", "mae",  # share pred/truth data, loaded once
    "overlap_at_20",
])
results = collection.evaluate(data, aggregation="logfc")
```

Automatically groups metrics sharing the same input data to avoid redundant computation.

### Wrappers (composable)

```python
CelltypeWise(metric="pearson_delta", celltype_key="cell_type")
Bootstrap(metric="mse", n_bootstraps=1000, ci=0.95)
DEGMask(metric="mse", n_top_genes=20)
```

Wrappers stack: `Bootstrap(CelltypeWise(DEGMask("mse")))` — bootstrap MSE on top-20 DE genes per cell type.

---

## 6. L4 — perteval.bench

### Evaluator (single model, atomic operation)

```python
class Evaluator:
    def evaluate(
        self,
        data: PerturbationData,
        metrics: list[str] | str = "default",  # "default" = [pearson_delta, mse, overlap_at_20]
        aggregation: str = "average",
    ) -> EvalResult:
```

### EvalResult (self-describing)

```python
@dataclass
class EvalResult:
    per_perturbation: pl.DataFrame   # perturbation × metric matrix
    aggregated: pl.DataFrame         # mean/std/min/max per metric
    config: dict                     # reproducibility: metrics, aggregation, timestamp, git hash

    def to_json(self, path: str) -> None: ...   # canonical format
    def to_csv(self, path: str) -> None: ...    # human-friendly export
```

### BenchmarkRunner (multi-model orchestration)

```python
runner = BenchmarkRunner(
    benchmarks=["norman19", "replogle22"],  # YAML names or paths
    models=["mean_control", "cpa", "gears"],
)
results: dict[str, dict[str, EvalResult]] = runner.run()
# results["norman19"]["cpa"] → EvalResult
```

Internal flow:
1. TaskManager scans `benchmarks/` directory, loads YAML definitions
2. For each (benchmark, model) pair: load data → split → train → predict → evaluate
3. Collect all EvalResult objects

### Compare

```python
comparison = Compare.from_results(results)
comparison.summary()  # multi-model × multi-dataset × multi-metric comparison table

robust = Compare.evaluate_many(runner, seeds=[0, 1, 2, 3, 4])
robust.summary()  # mean ± std per (model, dataset, metric)
```

### CLI

```bash
perteval run --benchmark norman19 --model cpa
perteval run --benchmark norman19 replogle22 --model cpa gears mean_control
perteval evaluate --predicted pred.h5ad --ground-truth real.h5ad --metrics pearson_delta mse
perteval list models
perteval list metrics
perteval list benchmarks
```

---

## 7. Benchmark YAML Schema

```yaml
# Required (v1)
dataset: norman19
metrics: [pearson_delta, mse]

# Optional with defaults
split:
  method: random              # random | transfer | combination (default: random)
  frac: [0.8, 0.1, 0.1]     # default: [0.8, 0.1, 0.1]
  holdout_key: null           # for transfer splits
  seed: 42
aggregation: average          # average | logfc | pca | de_scores (default: average)

# Optional: schema validation for input AnnData
schema:
  obs_required: [perturbation, cell_type]  # required columns in adata.obs
  var_required: [gene_name]                # required columns in adata.var
  layers: [X]                              # required layers

# Future extensions (all optional)
eval_dims: [per_pert]         # default: [per_pert]
deg_filter: null              # default: no filtering
baseline: null                # default: no baseline comparison
leaderboard: null             # default: no leaderboard generation
```

**Evolution rule:** New fields are always optional with sensible defaults. Existing YAML files never break.

**Schema validation:** When `schema` is present, data is validated at load time (before training/inference), giving early errors like "your AnnData is missing the `cell_type` column" instead of failing deep in the evaluation pipeline.

---

## 8. Configuration System

Simple YAML + Python API dual entry:

**YAML** — declarative, easy to read, defines "what to evaluate":
```yaml
dataset: norman19
metrics: [pearson_delta, mse, overlap_at_20]
split: {method: transfer, holdout_key: cell_type}
```

**Python** — programmatic, flexible, full control:
```python
evaluator = Evaluator()
data = PerturbationData(predicted=pred_adata, ground_truth=real_adata)
result = evaluator.evaluate(data, metrics=["pearson_delta", "mse"])
result.to_json("results.json")
```

No Hydra dependency. Complex experiment matrices handled via Python loops:
```python
for model_name in ["cpa", "gears", "mean_control"]:
    for benchmark in ["norman19", "replogle22"]:
        ...
```

---

## 9. Dependencies

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "anndata>=0.10",
    "numpy>=1.24",
    "polars>=0.20",
    "pyyaml>=6.0",
    "scipy>=1.11",
]

[project.optional-dependencies]
models = ["torch>=2.0", "scvi-tools>=1.0"]
data = ["datasets>=2.0"]
gears = ["gears"]
all = ["perteval[models,data,gears]"]
dev = ["pytest", "ruff", "ty"]
```

Core install (`pip install perteval`) is lightweight — only metrics layer dependencies. Model and data dependencies are opt-in.

---

## 10. Testing Strategy

**Unit tests (CI, fast):**
- Every metric, accessor, model wrapper tested independently
- Synthetic data via `build_random_anndata()` fixture — no external downloads
- Run on every push

**Integration tests (release, slow):**
- Full pipeline: load real data subset → train → predict → evaluate
- Small subsets of Norman19/Replogle22 committed as test fixtures
- Run before releases

**Tooling:** pytest + ruff + ty (type checker)

---

## 11. Project Structure

```
vc-perturb-eval/
├── pyproject.toml
├── src/perteval/
│   ├── __init__.py
│   ├── _registry.py            # Generic Registry[T] base class
│   ├── data/                   # L1
│   │   ├── accessors/
│   │   │   ├── base.py         # DataAccessor Protocol
│   │   │   ├── huggingface.py
│   │   │   ├── scperturb.py
│   │   │   └── local.py
│   │   ├── splitter.py
│   │   └── types.py            # PerturbationData
│   ├── models/                 # L2
│   │   ├── base.py             # PerturbationModel Protocol
│   │   ├── registry.py
│   │   └── baselines/
│   │       ├── mean_control.py
│   │       ├── cpa.py
│   │       └── gears.py
│   ├── metrics/                # L3 (core, minimum install)
│   │   ├── registry.py
│   │   ├── base.py
│   │   ├── collection.py
│   │   ├── functional/
│   │   │   ├── expression.py
│   │   │   ├── de.py
│   │   │   └── distribution.py
│   │   └── wrappers/
│   │       ├── bootstrap.py
│   │       ├── celltype_wise.py
│   │       └── deg_mask.py
│   ├── bench/                  # L4
│   │   ├── evaluator.py
│   │   ├── runner.py
│   │   ├── compare.py
│   │   ├── result.py
│   │   └── task_manager.py
│   └── cli/
│       └── main.py
├── benchmarks/                 # YAML benchmark definitions
│   ├── norman19.yaml
│   ├── replogle22.yaml
│   └── ...
├── tests/
│   ├── unit/
│   │   ├── test_metrics/
│   │   ├── test_data/
│   │   └── test_models/
│   ├── integration/
│   │   └── test_pipeline.py
│   └── conftest.py
└── docs/
```

---

## 12. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Target users | Self-use first, open-source ready (C) | Design for community from day one |
| Scope | Full pipeline, modular opt-in | Users choose which layers to use |
| Data format | AnnData core, MuData reserved | scverse ecosystem consensus |
| Inter-layer contract | Frozen dataclass holding AnnData refs | Validated interface without losing bio metadata |
| Model Protocol | load/train/predict (delegation) | Framework doesn't control training loops |
| Metric Registry | Lazy loading + entry_points plugins | Avoid heavy imports; enable third-party metrics |
| Benchmark YAML | dataset + split + metrics + aggregation | Reproducible "what to evaluate"; models in Python |
| Multi-model comparison | Evaluator (atomic) + BenchmarkRunner (orchestration) | Clean separation of concerns |
| Results | JSON canonical + CSV export | Reproducibility + human-friendliness |
| Config system | Simple YAML + Python API | Low onboarding cost, no Hydra dependency |
| Dependencies | Optional groups (models, data, all) | Core install is lightweight |
| Testing | Unit (synthetic) + integration (real subset) | Fast CI + thorough release validation |
| Python version | 3.11+ | Modern syntax, performance, target user environments |
| YAML evolution | New fields always optional + default | Never break existing configs |
| Training | Delegation to model.train() | Each model's training is too different to unify |

---

## 13. Borrowed Patterns (Attribution)

| Pattern | Source | Where used |
|---|---|---|
| Registry + Protocol metric interface | cell-eval | L3 metrics |
| Frozen dataclass as pipeline IR | cell-eval | L1 PerturbationData |
| 12 aggregation strategies | PerturBench | L4 aggregation options |
| Pairwise + ranking evaluation | PerturBench | L4 future extension |
| Multi-model dict comparison | PerturBench | L4 BenchmarkRunner |
| YAML task definitions + auto-discovery | lm-eval-harness | L4 TaskManager |
| Lazy Registry with string placeholders | lm-eval-harness | L2/L3 registries |
| Filter pipeline concept | lm-eval-harness | L4 future extension |
| update/compute/reset streaming | TorchMetrics | L3 OOP metrics |
| MetricCollection auto compute groups | TorchMetrics | L3 MetricCollection |
| Wrapper composition (Bootstrap, Classwise) | TorchMetrics | L3 wrappers |
| Functional/OOP dual API | TorchMetrics | L3 functional + OOP |
| BenchmarkGroup (dataset+metric+split) | TDC | L4 YAML benchmarks |
| evaluate_many (multi-seed robustness) | TDC | L4 Compare |
| DataAccessor pattern | TDC | L1 accessors |
| Schema validation for data contracts | OpenProblems | L1/YAML schema field |

---

## 14. Future Considerations (v2+)

Items explicitly deferred from v1. Listed here so they inform v1 interface design without adding implementation scope.

- **Matrix experiment YAML**: Declarative experiment matrix (benchmarks × models × aggregations × seeds) as a convenience layer over BenchmarkRunner. Pure addon — parses YAML into existing BenchmarkRunner constructor args.
- **Distributed evaluation**: Support for multi-GPU/multi-node metric computation (rank/world_size sharding, dist_reduce_fx). Not needed for v1 dataset sizes; model training already handles its own DDP.
- **Profile mechanism**: Named metric presets (e.g., `"minimal"`, `"full"`) defined via YAML, as aliases for metric lists. Pure addon over existing metrics list.
- **Aggregation registry**: If aggregation strategies grow beyond 3-4, extract from Evaluator into independent registry with `AggregationStrategy` Protocol.
- **Pairwise + ranking evaluation**: N×N perturbation pairwise metrics and ranking accuracy (PerturBench pattern).
- **Filter pipeline**: Post-prediction processing chain before metrics (lm-eval-harness pattern).
- **MuData support**: Multimodal data via MuData for multi-omics perturbation experiments.
