# Perturbation Prediction Benchmark Frameworks: Deep Comparison Report

> Review: Architectural comparison of 7 popular eval frameworks, distilling reusable design patterns
> Goal: design a unified evaluation framework for single-cell perturbation prediction 


---

## 1. Overview Matrix

| Dimension | cell-eval | PerturBench | scPerturBench | lm-eval-harness | TorchMetrics | OpenProblems | TDC |
|---|---|---|---|---|---|---|---|
| **Positioning** | Evaluation API | End-to-end benchmark | Method comparison study | LLM eval framework | Metric library | Community benchmark platform | Drug discovery benchmark |
| **Stars** | ~124 | ~77 | ~76 | ~12K | ~2.4K | ~18 | ~1.2K |
| **Architecture** | Registry + Pipeline | Hydra Config + Lightning | Script collection | YAML Registry + Factory | OOP base class | Viash components + Nextflow | Hierarchical API + BenchmarkGroup |
| **Configuration** | Python dataclass | Hydra YAML | Hardcoded | YAML + `!function` | Python constructor | YAML component declaration | Python + metadata dict |
| **Extensibility** | ★★★★ | ★★★ | ★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★ |
| **Onboarding cost** | Low | Medium | High | Medium | Low | High | Low |
| **Code quality** | Excellent | Good | Poor | Excellent | Excellent | Good | Fair |

---

## 2. In-Depth Analysis

### 1. cell-eval (Arc Institute) — The Gold Standard for Evaluation Layers

**Core architecture:**
- `MetricRegistry` singleton + `MetricPipeline` + `MetricsEvaluator`
- Metrics defined via Protocol (duck typing), no mandatory base class inheritance
- Profile mechanism (`full|minimal|vcc|de|anndata`) controls which metrics to run

**Pros:**
- **Clean registry pattern**: `metrics_registry.register(name, type, desc, func)` — single-line registration
- **Profile-based selection**: Users pick a profile instead of cherry-picking individual metrics
- **Strong type system**: Immutable data classes (`PerturbationAnndataPair`, `DEComparison`) prevent accidental mutation
- **Well-designed CLI**: `prep → run → baseline → score` — clear four-step workflow
- **Protocol-based metrics**: Functions or classes can both be registered, zero inheritance overhead

**Cons:**
- **Evaluation only, no model inference**: Users must produce prediction AnnData on their own
- **Profiles are hardcoded**: Cannot define new profiles via configuration files
- **No multi-dataset batching**: One dataset per evaluation run

**Takeaways:**
> ✅ MetricRegistry registration pattern + Profile mechanism
> ✅ Protocol-based metric interface (no forced inheritance)
> ✅ Immutable data classes as pipeline intermediate representations

---

### 2. PerturBench (Altos Labs) — The Complete End-to-End Solution

**Core architecture:**
- Hydra config system drives everything (data + model + trainer + experiment)
- `PerturbationModel(LightningModule)` base class + 7 model implementations
- `Evaluation` object manages: prediction → aggregation → evaluation → ranking
- Accessor pattern for dataset loading (HuggingFace / LaminDB support)

**Pros:**
- **Powerful Hydra composition**: `experiment=neurips2025/replogle` switches the entire experiment setup in one line
- **Training-evaluation integration**: `on_test_end` auto-triggers evaluation, no extra steps
- **Rich aggregation strategies**: logFC, scaled, PCA, and more for perturbation-level aggregation
- **Well-designed Evaluation object**: `aggregate() → evaluate() → evaluate_rank()` chained calls
- **Reproducible training records**: `training_record` stores transforms and context

**Cons:**
- **Tight training-evaluation coupling**: Running evaluation-only (without training) is difficult
- **High barrier for new models**: Must understand full Lightning stack — inherit `PerturbationModel`, write Hydra config, understand batch format
- **No independent metric registry**: Metrics are hardcoded in `compute_metric()`
- **Missing tests**: No test suite found

**Takeaways:**
> ✅ Hydra compositional config for experiment management (dataset × model × hyperparams)
> ✅ Evaluation object's aggregate → evaluate → rank pipeline
> ✅ Accessor pattern decoupling data sources (HuggingFace / local / LaminDB)

---

### 3. scPerturBench (bm2-lab) — The Anti-Pattern

**Core architecture:**
- No architecture. 58 Python scripts organized by scenario directories
- Each method = standalone script (120-440 LOC), no shared interface
- `myUtil.py` mixes preprocessing, DEG computation, and data loading
- `calPerformance.py` uses pertpy.tools.Distance for metric computation

**Pros:**
- **Simple and direct**: No framework to learn, just read and run scripts
- **Broad coverage**: 27 methods × 29 datasets, rich results
- **Podman containers**: Solves dependency hell (9 conda environments packaged)
- **pertpy for metrics**: Leverages a mature library for E-distance, Wasserstein, etc.

**Cons:**
- **Massive code duplication**: 80% identical code across method scripts (data loading, splitting, output)
- **Hardcoded paths**: `/home/project/Pertb_benchmark/DataSet/` baked in
- **Not pip-installable**: No Python package, no `__init__.py`
- **Adding a new method = copy-paste template**: Not extensible
- **calPerformance duplicated across scenarios**: Genetic and chemical versions share 80% code

**Takeaways:**
> ✅ pertpy.tools.Distance as the metric backend (mature, comprehensive)
> ✅ DEG-based evaluation subset strategy (top-100 / top-5000)
> ⚠️ Serves as an anti-pattern: this is exactly the problem good architecture should solve

---

### 4. lm-eval-harness (EleutherAI) — The Gold Standard for Extensibility

**Core architecture:**
- **Two-phase lazy loading**: TaskIndex (lightweight YAML scan) → TaskFactory (on-demand instantiation)
- **Three independent registries**: Task, Model, Metric — each with its own registration
- **ConfigurableTask**: YAML config maps to task behavior, `!function` tags reference Python functions
- **CachingLM**: SQLite caches model responses to avoid redundant inference
- **Abstract LM interface**: `loglikelihood()` + `generate_until()` + `loglikelihood_rolling()`

**Pros:**
- **Zero-code task definition via YAML**: Non-developers can contribute new benchmark scenarios
- **Registry + Factory separation**: Discovery (what exists) and instantiation (how to create) are decoupled
- **Multi-level caching**: Request-level (pickle), model-response-level (SQLite), module-level (importlib mtime)
- **`!function` bridge**: Perfect blend of YAML declarative style + Python flexibility
- **Composable**: Task Groups nest, Tags filter, Filters post-process

**Cons:**
- **LLM-specific abstractions**: loglikelihood / generate_until don't apply to perturbation prediction
- **High complexity**: Understanding the full architecture requires reading 10+ files
- **YAML debugging is painful**: Error messages for broken `!function` references are poor

**Takeaways:**
> ✅ Two-phase lazy loading (Index lightweight scan → Factory on-demand creation)
> ✅ YAML + `!function` pattern (declarative config + Python extension points)
> ✅ Registry + Factory separation pattern
> ✅ Multi-level caching strategy

---

### 5. TorchMetrics (Lightning AI) — The Gold Standard for Metric Design

**Core architecture:**
- `Metric` base class: `add_state()` declarative state + `update()` accumulate + `compute()` calculate
- `MetricCollection`: Smart composition, auto-detects metrics sharing state and merges into Compute Groups
- Automatic distributed sync: `sync_context()` transparently gathers + reduces during `compute()`
- Functional API: Stateless pure functions; Class API internally delegates to Functional

**Pros:**
- **Declarative state management**: `add_state("tp", tensor(0), dist_reduce_fx="sum")` — one line for state + reduction
- **Fully transparent DDP**: User metric code has zero distributed logic; base class handles everything
- **Compute Group optimization**: Precision/Recall/F1 share TP/FP/TN/FN, updated only once
- **Wrapper pattern**: BootStrapper (confidence intervals), MetricTracker (cross-epoch tracking)
- **Dual API (Functional + Class)**: Flexibility + state management combined

**Cons:**
- **Hard PyTorch binding**: States must be Tensors, unsuitable for DataFrame-based biological metrics
- **Not a benchmark framework**: Only handles metric computation, not data loading, model inference, or result aggregation
- **Learning curve**: `add_state` + `dist_reduce_fx` + sync/unsync concepts are unfriendly to newcomers

**Takeaways:**
> ✅ `add_state() + update() + compute()` three-phase metric lifecycle
> ✅ MetricCollection's Compute Group optimization concept
> ✅ Wrapper pattern (BootStrapper / MetricTracker)
> ⚠️ Adaptation needed: our metrics take AnnData/DataFrame, not Tensors

---

### 6. OpenProblems task_perturbation_prediction — Most Modular but Heaviest

**Core architecture:**
- Viash component system: each method/metric = YAML declaration + script + Docker image
- Shared API definitions: `comp_method.yaml` (method template), `comp_metric.yaml` (metric template)
- Nextflow DSL2 orchestration: `run_benchmark` workflow chains method → metric → aggregate
- File I/O communication: components exchange data via h5ad/csv files

**Pros:**
- **Language-agnostic**: Python, R, Shell can all implement methods/metrics
- **Declarative interfaces**: YAML defines input/output types, validated at compile time
- **Full isolation**: Each component gets its own Docker image, zero dependency conflicts
- **Community-friendly**: `add_a_method.sh` scaffolding script
- **CI/CD integration**: GitHub Actions + AWS automated benchmark runs

**Cons:**
- **Heavy toolchain**: Requires Viash + Nextflow + Docker, high barrier to entry
- **Poor developer experience**: Local debugging needs `## VIASH START/END` blocks, unintuitive
- **File I/O overhead**: Components communicate via disk, slow with large datasets
- **Hardcoded method lists**: `run_benchmark/main.nf` manually lists all methods
- **Over-engineered**: For running benchmarks, three layers of abstraction (Nextflow + Viash + Docker) is excessive

**Takeaways:**
> ✅ Shared API templates (comp_method.yaml / comp_metric.yaml) defining interface contracts
> ✅ Scaffolding scripts (add_a_method.sh) lowering contribution barriers
> ⚠️ Container isolation concept is sound, but Viash+Nextflow is too heavy — use a lighter approach

---

### 7. TDC (Harvard) — Best API Ergonomics

**Core architecture:**
- Three-tier hierarchy: Problem → Task → Dataset, code structure mirrors directory layout
- `BenchmarkGroup`: Bundles dataset + split + evaluator, iterator pattern
- `Evaluator`: Fuzzy-matches metric name → factory dispatches to concrete implementation
- Data download + local cache, transparent data acquisition

**Pros:**
- **Minimal user experience**: 3 lines of code for a full benchmark (load → split → evaluate)
- **Fuzzy matching**: `Evaluator("roc-auc")` tolerates typos
- **BenchmarkGroup iterator**: `for name, data in group` auto-traverses all sub-datasets
- **Multiple split strategies**: random / cold / scaffold / temporal — domain-expert-level design
- **Flexible formats**: `get_data(format="df"|"dict"|"DeepPurpose")`

**Cons:**
- **Giant metadata file**: 900+ line metadata.py, adding new datasets requires modifying this file
- **Hardcoded Evaluator**: Factory uses if-else dispatch, not a registry pattern
- **Weak typing**: Heavy use of `Any` and implicit types
- **Centralized data hosting**: Harvard Dataverse, single point of failure risk

**Takeaways:**
> ✅ BenchmarkGroup's "dataset + split + evaluator" bundling pattern
> ✅ 3-line benchmark as the API ergonomics target
> ✅ Multiple domain-aware split strategies (scaffold / cold-start)
> ⚠️ Metadata management needs improvement (use YAML + registry instead of giant dicts)

---

## 3. Cross-Cutting Comparison: Key Design Decisions

### 3.1 Metric System Design

| Approach | Representative | Registration | Input Type | State Mgmt | Distributed |
|---|---|---|---|---|---|
| Protocol + Registry | cell-eval | `registry.register()` | AnnData pair | Stateless | No |
| Base class + add_state | TorchMetrics | Inherit `Metric` | Tensor | Declarative | Auto DDP |
| Decorator registration | lm-eval-harness | `@register_metric` | dict | Stateless | No |
| Hardcoded functions | PerturBench/TDC | if-else dispatch | DataFrame | Stateless | No |
| pertpy delegation | scPerturBench | Direct calls | AnnData | Stateless | No |

**Our best approach:** cell-eval's Registry + Protocol pattern, combined with TorchMetrics' `update/compute` lifecycle concept (but not bound to Tensors — adapted for AnnData/DataFrame).

### 3.2 Configuration System

| Approach | Representative | Pros | Cons |
|---|---|---|---|
| Hydra YAML | PerturBench | Powerful composition, flexible overrides | Steep learning curve, hard to debug |
| Custom YAML + `!function` | lm-eval-harness | Declarative + extensible | Must maintain custom loader |
| Python dataclass | cell-eval | Type-safe, IDE-friendly | Not suitable for non-developers |
| Viash YAML | OpenProblems | Strict interfaces, language-agnostic | Heavy toolchain |
| metadata dict | TDC | Simple and direct | Not extensible |

**Our best approach:** Hydra YAML (mature ecosystem) for experiment configuration, Python dataclasses for internal types.

### 3.3 Model Interface

| Approach | Representative | Onboarding Cost | Framework Assumptions |
|---|---|---|---|
| Inherit LightningModule | PerturBench | High (must know Lightning) | Strong (training framework lock-in) |
| Implement 3 methods | lm-eval-harness | Medium | Medium (LM abstractions) |
| File I/O contract | OpenProblems | Low (read h5ad, write h5ad) | None (language-agnostic) |
| No interface | scPerturBench | Lowest (copy script) | None |

**Our best approach:** Define a lightweight Protocol (no forced inheritance), core method: `predict(control_adata, perturbations) → predicted_adata`. Also support file I/O mode (submit h5ad directly).

### 3.4 Data Standardization

| Approach | Representative | Standard Format | Data Sources |
|---|---|---|---|
| Direct AnnData | cell-eval, scPerturBench | h5ad | Local files |
| Accessor + DataModule | PerturBench | h5ad → NamedTuple | HuggingFace / LaminDB |
| Viash file contract | OpenProblems | h5ad (schema-defined) | S3 |
| Dataverse download | TDC | CSV/PKL | Harvard Dataverse |

**Our best approach:** AnnData (h5ad) as the standard exchange format (domain consensus), Accessor pattern for multiple data sources, YAML schema defining obs/var/layers contracts.

---

## 4. Top 10 Design Patterns to Adopt

| # | Pattern | Source | Why |
|---|---|---|---|
| 1 | **Registry + Protocol metrics** | cell-eval | Zero-inheritance metric registration; functions or classes both work |
| 2 | **YAML task declaration + lazy loading** | lm-eval-harness | Non-developers can define new benchmark scenarios |
| 3 | **Hydra compositional config** | PerturBench | `dataset=X model=Y metric=Z` defines an experiment in one line |
| 4 | **BenchmarkGroup bundling** | TDC | Dataset + split + evaluator atomized; 3 lines to run |
| 5 | **update/compute lifecycle** | TorchMetrics | Metrics are accumulative, composable, and trackable |
| 6 | **Compute Group optimization** | TorchMetrics | Metrics sharing state are merged for computation |
| 7 | **API templates (interface contracts)** | OpenProblems | YAML defines method/metric input-output types |
| 8 | **Evaluation chain pipeline** | PerturBench | predict → aggregate → evaluate → rank |
| 9 | **Profile-based metric selection** | cell-eval | Users pick a profile rather than individual metrics |
| 10 | **DEG-based evaluation subsets** | scPerturBench | top-100/5000 DEG focus on biological significance |

---

## 5. Proposed Architecture Blueprint

Synthesizing all analyses, our framework should look like this:

```
User perspective (3 lines of code):
  benchmark = PertBench("norman2019", split="cold_perturbation")
  results = benchmark.evaluate(my_model)  
  results.to_leaderboard()

Internal layers:
  ┌─────────────────────────────────────────────┐
  │  Config Layer (Hydra YAML)                  │  ← Inspired by PerturBench
  │  dataset=X  model=Y  metrics=[Z1,Z2]       │
  ├─────────────────────────────────────────────┤
  │  Task Registry (YAML + lazy load)           │  ← Inspired by lm-eval-harness
  │  Each benchmark scenario = one YAML file    │
  ├─────────────────────────────────────────────┤
  │  Data Layer                                 │
  │  Accessor (HF/local/S3) → AnnData (h5ad)   │  ← Inspired by PerturBench + OpenProblems
  │  Schema validation (obs/var/layers contract)│
  ├─────────────────────────────────────────────┤
  │  Model Adapter (Protocol-based)             │  ← Inspired by cell-eval
  │  predict(control, perturbations) → adata    │
  │  Also supports direct h5ad file submission  │
  ├─────────────────────────────────────────────┤
  │  Evaluation Pipeline                        │  ← Inspired by PerturBench + cell-eval
  │  aggregate() → evaluate() → rank()          │
  ├─────────────────────────────────────────────┤
  │  Metric Layer                               │
  │  Registry (Protocol) + Profile selection    │  ← Inspired by cell-eval + TorchMetrics
  │  update/compute lifecycle                   │
  │  pertpy backend                             │  ← Inspired by scPerturBench
  ├─────────────────────────────────────────────┤
  │  BenchmarkGroup                             │  ← Inspired by TDC
  │  dataset + split + metrics bundled          │
  │  Iterator pattern for batch evaluation      │
  └─────────────────────────────────────────────┘
```

---

## 6. Anti-Patterns to Avoid

| Anti-Pattern | Source | Lesson |
|---|---|---|
| Hardcoded paths | scPerturBench | All paths must be configurable |
| Giant metadata dict | TDC | Use YAML + registry instead |
| Tight training-evaluation coupling | PerturBench | Evaluation must run independently |
| Viash + Nextflow + Docker triple stack | OpenProblems | Keep the toolchain light; pip install should be enough |
| if-else metric dispatch | TDC / PerturBench | Use a registry pattern |
| Copy-paste template for new methods | scPerturBench | Use Protocol + scaffolding |
