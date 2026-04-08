# Perturbation Prediction Benchmark Frameworks: Deep Comparison Report

> 7 个开源项目的架构深度对比，提炼可借鉴的设计模式
> Generated: 2026-04-07

---

## 一、总览矩阵

| 维度 | cell-eval | PerturBench | scPerturBench | lm-eval-harness | TorchMetrics | OpenProblems | TDC |
|---|---|---|---|---|---|---|---|
| **定位** | 评估 API | 端到端 benchmark | 方法对比研究 | LLM 评测框架 | Metric 库 | 社区 benchmark 平台 | 药物发现 benchmark |
| **Stars** | ~124 | ~77 | ~76 | ~12K | ~2.4K | ~18 | ~1.2K |
| **架构风格** | Registry + Pipeline | Hydra Config + Lightning | 脚本集合 | YAML Registry + Factory | OOP 基类 | Viash 组件 + Nextflow | 层级 API + BenchmarkGroup |
| **配置方式** | Python dataclass | Hydra YAML | 硬编码 | YAML + `!function` | Python 构造函数 | YAML 组件声明 | Python + metadata dict |
| **可扩展性** | ★★★★ | ★★★ | ★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★ |
| **用户上手成本** | 低 | 中 | 高 | 中 | 低 | 高 | 低 |
| **代码质量** | 优 | 良 | 差 | 优 | 优 | 良 | 中 |

---

## 二、逐项深度分析

### 1. cell-eval (Arc Institute) — 评估层的典范

**架构核心：**
- `MetricRegistry` 单例 + `MetricPipeline` + `MetricsEvaluator`
- Metric 通过 Protocol（鸭子类型）定义，不强制继承基类
- Profile 机制（`full|minimal|vcc|de|anndata`）控制跑哪些 metric

**Pros:**
- **Registry 模式干净**：`metrics_registry.register(name, type, desc, func)` 一行注册
- **Profile-based 选择**：用户不需要逐个选 metric，选个 profile 就行
- **类型系统好**：`PerturbationAnndataPair`、`DEComparison` 等不可变数据类，防止意外修改
- **CLI 设计合理**：`prep → run → baseline → score` 四步流程清晰
- **Protocol-based metric**：函数或类都能注册，零继承负担

**Cons:**
- **只做评估，不管模型推理**：用户要自己搞定预测结果的 AnnData
- **Profile 硬编码**：不能通过配置文件定义新 profile
- **不支持多数据集批量跑**：一次只能评估一个数据集

**可借鉴：**
> ✅ MetricRegistry 的注册模式 + Profile 机制
> ✅ Protocol-based metric 接口（不强制继承）
> ✅ 不可变数据类型作为 pipeline 中间表示

---

### 2. PerturBench (Altos Labs) — 端到端的完整方案

**架构核心：**
- Hydra 配置系统驱动一切（data + model + trainer + experiment）
- `PerturbationModel(LightningModule)` 基类 + 7 个模型实现
- `Evaluation` 对象管理：预测 → 聚合 → 评估 → 排名 全流程
- Accessor 模式加载数据集（支持 HuggingFace / LaminDB）

**Pros:**
- **Hydra 组合配置极强**：`experiment=neurips2025/replogle` 一行切换完整实验设置
- **训练-评估一体化**：`on_test_end` 自动触发评估，无需额外步骤
- **聚合策略丰富**：logFC、scaled、PCA 等多种 perturbation-level 聚合方式
- **Evaluation 对象设计好**：`aggregate() → evaluate() → evaluate_rank()` 链式调用
- **训练记录可追溯**：`training_record` 存储 transforms 和 context

**Cons:**
- **训练和评估强耦合**：想只跑评估（不训练）比较困难
- **添加新模型需要理解 Lightning 全套**：继承 `PerturbationModel`、写 Hydra config、理解 batch 格式
- **没有独立的 metric registry**：metric 在 `compute_metric()` 里硬编码
- **缺少测试**：未找到测试套件

**可借鉴：**
> ✅ Hydra 组合配置管理实验（数据集 × 模型 × 超参）
> ✅ Evaluation 对象的 aggregate → evaluate → rank 流程
> ✅ Accessor 模式解耦数据来源（HuggingFace / 本地 / LaminDB）

---

### 3. scPerturBench (bm2-lab) — 反面教材

**架构核心：**
- 没有架构。58 个 Python 脚本，按场景目录组织
- 每个方法 = 一个独立脚本（120-440 LOC），无共享接口
- `myUtil.py` 混合了预处理、DEG 计算、数据加载
- `calPerformance.py` 用 pertpy.tools.Distance 计算指标

**Pros:**
- **简单直接**：不需要学任何框架，看脚本就能跑
- **覆盖面广**：27 个方法 × 29 个数据集，结果丰富
- **Podman 容器**：解决了依赖地狱（9 个 conda 环境打包）
- **pertpy 做 metric**：利用成熟库计算 E-distance、Wasserstein 等

**Cons:**
- **大量重复代码**：每个方法脚本 80% 代码相同（数据加载、split、输出）
- **硬编码路径**：`/home/project/Pertb_benchmark/DataSet/` 写死
- **无法 pip install**：不是 Python 包，无 `__init__.py`
- **添加新方法 = 复制粘贴模板**：不可扩展
- **calPerformance 场景间 80% 重复**：基因和化合物各写一份

**可借鉴：**
> ✅ pertpy.tools.Distance 作为 metric backend 的选择（成熟、全面）
> ✅ DEG-based 评估子集选择策略（top-100 / top-5000）
> ⚠️ 作为反面教材：这就是我们要用好架构解决的问题

---

### 4. lm-eval-harness (EleutherAI) — 可扩展性的金标准

**架构核心：**
- **两阶段懒加载**：TaskIndex（轻量扫描 YAML）→ TaskFactory（按需实例化）
- **三大 Registry**：Task、Model、Metric 各自独立注册
- **ConfigurableTask**：YAML 映射到 Task 行为，`!function` 标签引用 Python 函数
- **CachingLM**：SQLite 缓存模型响应，避免重复推理
- **抽象 LM 接口**：`loglikelihood()` + `generate_until()` + `loglikelihood_rolling()`

**Pros:**
- **YAML 定义 Task 零代码**：非开发者也能贡献新 benchmark
- **Registry + Factory 分离**：发现（what exists）和实例化（how to create）解耦
- **多级缓存**：请求级（pickle）、模型响应级（SQLite）、模块级（importlib mtime）
- **`!function` 桥接**：YAML 声明式 + Python 灵活性完美结合
- **可组合**：Task Group 嵌套、Tag 过滤、Filter 后处理

**Cons:**
- **LLM 特定抽象**：loglikelihood / generate_until 不适用于 perturbation prediction
- **复杂度高**：理解完整架构需要读 10+ 文件
- **YAML 调试困难**：`!function` 引用出错时报错信息差

**可借鉴：**
> ✅ 两阶段懒加载（Index 轻量扫描 → Factory 按需创建）
> ✅ YAML + `!function` 模式（声明式配置 + Python 扩展点）
> ✅ Registry + Factory 分离模式
> ✅ 多级缓存策略

---

### 5. TorchMetrics (Lightning AI) — Metric 层的金标准

**架构核心：**
- `Metric` 基类：`add_state()` 声明式状态 + `update()` 累积 + `compute()` 计算
- `MetricCollection`：智能组合，自动检测共享状态的 metric 合并为 Compute Group
- 自动分布式同步：`sync_context()` 在 `compute()` 时透明 gather + reduce
- Functional API：无状态纯函数，Class API 内部委托给 Functional

**Pros:**
- **声明式状态管理**：`add_state("tp", tensor(0), dist_reduce_fx="sum")` 一行搞定状态+归约
- **DDP 完全透明**：用户写的 metric 代码零分布式逻辑，base class 自动处理
- **Compute Group 优化**：Precision/Recall/F1 共享 TP/FP/TN/FN，只 update 一次
- **Wrapper 模式**：BootStrapper（置信区间）、MetricTracker（跨 epoch 追踪）
- **Functional + Class 双 API**：灵活性 + 状态管理兼得

**Cons:**
- **PyTorch 强绑定**：状态必须是 Tensor，不适合 DataFrame-based 的生物指标
- **不是 benchmark 框架**：只管 metric 计算，不管数据加载、模型推理、结果聚合
- **学习曲线**：`add_state` + `dist_reduce_fx` + sync/unsync 概念对新手不友好

**可借鉴：**
> ✅ `add_state() + update() + compute()` 三段式 metric 生命周期
> ✅ MetricCollection 的 Compute Group 优化思路
> ✅ Wrapper 模式（BootStrapper / MetricTracker）
> ⚠️ 但需要适配：我们的 metric 输入是 AnnData/DataFrame，不是 Tensor

---

### 6. OpenProblems task_perturbation_prediction — 最模块化但最重

**架构核心：**
- Viash 组件系统：每个 method/metric = YAML 声明 + 脚本 + Docker 镜像
- 共享 API 定义：`comp_method.yaml`（方法模板）、`comp_metric.yaml`（指标模板）
- Nextflow DSL2 编排：`run_benchmark` workflow 串联 method → metric → aggregate
- 文件 I/O 通信：组件间通过 h5ad/csv 文件交换数据

**Pros:**
- **语言无关**：Python、R、Shell 都能写 method/metric
- **声明式接口**：YAML 定义输入/输出类型，编译时校验
- **完全隔离**：每个组件独立 Docker，零依赖冲突
- **社区友好**：`add_a_method.sh` 脚手架脚本
- **CI/CD 集成**：GitHub Actions + AWS 自动化跑 benchmark

**Cons:**
- **工具链重**：需要安装 Viash + Nextflow + Docker，门槛高
- **开发体验差**：本地调试要 `## VIASH START/END` 块，不直观
- **文件 I/O 开销**：组件间通过磁盘传数据，大数据集慢
- **方法列表硬编码**：`run_benchmark/main.nf` 里手动列出所有方法
- **过度工程**：对于跑个 benchmark 来说，Nextflow + Viash + Docker 三层抽象太多

**可借鉴：**
> ✅ 共享 API 模板（comp_method.yaml / comp_metric.yaml）定义接口契约
> ✅ 脚手架脚本（add_a_method.sh）降低贡献门槛
> ⚠️ 容器隔离的思路好，但 Viash+Nextflow 太重，可以用更轻量的方案

---

### 7. TDC (Harvard) — API 人体工学最佳

**架构核心：**
- 三层层级：Problem → Task → Dataset，代码结构映射目录
- `BenchmarkGroup`：打包 dataset + split + evaluator，迭代器模式
- `Evaluator`：fuzzy match metric 名 → factory 分发到具体实现
- 数据下载 + 本地缓存，透明化数据获取

**Pros:**
- **用户体验极简**：3 行代码跑完整 benchmark（load → split → evaluate）
- **Fuzzy matching**：`Evaluator("roc-auc")` 容错拼写
- **BenchmarkGroup 迭代器**：`for name, data in group` 自动遍历所有子数据集
- **多种 split 策略**：random / cold / scaffold / temporal，领域专家级设计
- **格式灵活**：`get_data(format="df"|"dict"|"DeepPurpose")`

**Cons:**
- **metadata.py 巨型文件**：900+ 行配置，添加新数据集要改这个文件
- **Evaluator 硬编码**：factory 里 if-else 分发，不是 registry 模式
- **缺少类型标注**：大量 `Any` 和隐式类型
- **数据集中心化**：Harvard Dataverse 托管，单点故障风险

**可借鉴：**
> ✅ BenchmarkGroup 的「数据集 + split + evaluator」打包模式
> ✅ 3 行代码跑 benchmark 的 API 人体工学目标
> ✅ 多种领域感知的 split 策略（scaffold / cold-start）
> ⚠️ 但 metadata 管理方式需改进（用 YAML + registry 替代巨型 dict）

---

## 三、横向对比：关键设计决策

### 3.1 Metric 系统设计

| 方案 | 代表 | 注册方式 | 输入类型 | 状态管理 | 分布式 |
|---|---|---|---|---|---|
| Protocol + Registry | cell-eval | `registry.register()` | AnnData pair | 无状态 | 不支持 |
| 基类 + add_state | TorchMetrics | 继承 `Metric` | Tensor | 声明式 | 自动 DDP |
| 装饰器注册 | lm-eval-harness | `@register_metric` | dict | 无状态 | 不支持 |
| 硬编码函数 | PerturBench/TDC | if-else 分发 | DataFrame | 无状态 | 不支持 |
| pertpy 委托 | scPerturBench | 直接调用 | AnnData | 无状态 | 不支持 |

**我们的最佳方案：** cell-eval 的 Registry + Protocol 模式，加上 TorchMetrics 的 `update/compute` 生命周期概念（但不绑定 Tensor，改用 AnnData/DataFrame）。

### 3.2 配置系统

| 方案 | 代表 | 优点 | 缺点 |
|---|---|---|---|
| Hydra YAML | PerturBench | 组合能力强，override 灵活 | 学习曲线陡，调试难 |
| 自定义 YAML + `!function` | lm-eval-harness | 声明式+可扩展 | 自己维护 loader |
| Python dataclass | cell-eval | 类型安全，IDE 友好 | 不适合非开发者 |
| Viash YAML | OpenProblems | 接口严格，语言无关 | 工具链重 |
| metadata dict | TDC | 简单直接 | 不可扩展 |

**我们的最佳方案：** Hydra YAML（已有成熟生态）用于实验配置，Python dataclass 用于内部类型。

### 3.3 模型接口

| 方案 | 代表 | 接入成本 | 框架假设 |
|---|---|---|---|
| 继承 LightningModule | PerturBench | 高（需懂 Lightning） | 强（训练框架绑定） |
| 实现 3 个方法 | lm-eval-harness | 中 | 中（LM 抽象） |
| 文件 I/O 契约 | OpenProblems | 低（读 h5ad，写 h5ad） | 无（语言无关） |
| 无接口 | scPerturBench | 最低（复制脚本） | 无 |

**我们的最佳方案：** 定义一个轻量 Protocol（不强制继承），核心方法：`predict(control_adata, perturbations) → predicted_adata`。同时支持文件 I/O 模式（提交 h5ad 即可）。

### 3.4 数据标准化

| 方案 | 代表 | 标准格式 | 数据来源 |
|---|---|---|---|
| AnnData 直接使用 | cell-eval, scPerturBench | h5ad | 本地文件 |
| Accessor + DataModule | PerturBench | h5ad → NamedTuple | HuggingFace / LaminDB |
| Viash 文件契约 | OpenProblems | h5ad（schema 定义） | S3 |
| Dataverse 下载 | TDC | CSV/PKL | Harvard Dataverse |

**我们的最佳方案：** AnnData (h5ad) 作为标准交换格式（领域共识），Accessor 模式支持多数据来源，YAML schema 定义 obs/var/layers 契约。

---

## 四、我们应该借鉴的设计模式 Top 10

| # | 模式 | 来源 | 为什么 |
|---|---|---|---|
| 1 | **Registry + Protocol metric** | cell-eval | 零继承注册 metric，函数或类皆可 |
| 2 | **YAML task 声明 + 懒加载** | lm-eval-harness | 非开发者也能定义新 benchmark 场景 |
| 3 | **Hydra 组合配置** | PerturBench | `dataset=X model=Y metric=Z` 一行定义实验 |
| 4 | **BenchmarkGroup 打包** | TDC | 数据集+split+evaluator 原子化，3 行跑完 |
| 5 | **update/compute 生命周期** | TorchMetrics | metric 可累积、可组合、可追踪 |
| 6 | **Compute Group 优化** | TorchMetrics | 共享状态的 metric 合并计算 |
| 7 | **API 模板（接口契约）** | OpenProblems | YAML 定义 method/metric 的输入输出类型 |
| 8 | **Evaluation 链式流程** | PerturBench | predict → aggregate → evaluate → rank |
| 9 | **Profile-based metric 选择** | cell-eval | 用户选 profile 而非逐个选 metric |
| 10 | **DEG-based 评估子集** | scPerturBench | top-100/5000 DEG 聚焦生物学意义 |

---

## 五、建议的架构蓝图

综合以上分析，我们的框架应该是这样的：

```
用户视角（3 行代码）:
  benchmark = PertBench("norman2019", split="cold_perturbation")
  results = benchmark.evaluate(my_model)  
  results.to_leaderboard()

内部分层:
  ┌─────────────────────────────────────────────┐
  │  Config Layer (Hydra YAML)                  │  ← PerturBench 启发
  │  dataset=X  model=Y  metrics=[Z1,Z2]       │
  ├─────────────────────────────────────────────┤
  │  Task Registry (YAML + lazy load)           │  ← lm-eval-harness 启发
  │  每个 benchmark scenario = 一个 YAML 文件     │
  ├─────────────────────────────────────────────┤
  │  Data Layer                                 │
  │  Accessor (HF/local/S3) → AnnData (h5ad)   │  ← PerturBench + OpenProblems
  │  Schema validation (obs/var/layers 契约)     │
  ├─────────────────────────────────────────────┤
  │  Model Adapter (Protocol-based)             │  ← cell-eval 启发
  │  predict(control, perturbations) → adata    │
  │  也支持直接提交 h5ad 文件                      │
  ├─────────────────────────────────────────────┤
  │  Evaluation Pipeline                        │  ← PerturBench + cell-eval
  │  aggregate() → evaluate() → rank()          │
  ├─────────────────────────────────────────────┤
  │  Metric Layer                               │
  │  Registry (Protocol) + Profile 选择          │  ← cell-eval + TorchMetrics
  │  update/compute 生命周期                      │
  │  pertpy backend                             │  ← scPerturBench
  ├─────────────────────────────────────────────┤
  │  BenchmarkGroup                             │  ← TDC 启发
  │  dataset + split + metrics 打包              │
  │  iterator 模式批量评估                        │
  └─────────────────────────────────────────────┘
```

---

## 六、不应该做的事

| 反模式 | 来源 | 教训 |
|---|---|---|
| 硬编码路径 | scPerturBench | 路径必须可配置 |
| 巨型 metadata dict | TDC | 用 YAML + registry 替代 |
| 训练-评估强耦合 | PerturBench | 评估必须能独立运行 |
| Viash + Nextflow + Docker 三层 | OpenProblems | 工具链要轻，pip install 就能用 |
| if-else metric 分发 | TDC / PerturBench | 用 registry 模式 |
| 复制粘贴模板加新方法 | scPerturBench | 用 Protocol + 脚手架 |
