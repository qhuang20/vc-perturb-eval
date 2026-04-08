# 5 项目功能模块对比

## 核心功能矩阵

| 能力             | cell-eval                                    | PerturBench                                                                  | lm-eval-harness                        | TorchMetrics                | TDC                                    |
|------------------|----------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------|-----------------------------|----------------------------------------|
| Registry/注册机制 | Registry 单例 + Metric 注册，惰性加载          | ❌ 硬编码在 Evaluator 类 + compute_metric()，装饰器+模糊匹配                  | Registry + OOP 基类/Protocol 鸭子类型  | Metric 注册+add_state()      | ❌ 硬编码                              |
| Profile/预设任务 | Profile 预设（full/minimal/vcc...）           | -                                                                           | YAML 配置+metadata.py 声明式映射        | MetricCollection 自动分组     | 手动指定                              |
| 数据抽象         | 不可变 PerturbationInstance (prompt+target)   | AnnData dict keyed by model state/get_split()                                | DataFrame + keyed by model/get         | Tensor                       | -                                     |
| 聚合策略         | describe() (mean/std/percentile)              | 12 种聚合策略 (logfc/pca/de_scores...)                                      | evaluate_many (≥5 runs mean±std)       | update/compute 流式           | stderr/mean±std                       |
| 多模型对比       | ❌ 单次 pred vs real                          | ✅ dict keyed by model name                                                  | ✅ 多 model backend, N/A (metric 库)    | ❌ 单次                        | ❌                                    |
| Pairwise/Ranking | ❌                                           | ✅ N×N pairwise + ranking                                                    | ❌                                    | ❌                              | ❌                                    |
| 配置系统         | CLI args + Python API                        | Hydra YAML 组合，metadata.py 字典定义                                        | YAML task 构造器                      | -                             | -                                     |
| Benchmark 打包   | BenchmarkTaskManager 扫描目录自动注册         | ❌                                                                          | 5 个固定数据集+BenchmarkGroup(dataset+metrics+split) | -         | -                                     |
| 分布式/分片      | rank/world_size 原生支持                      | dist_reduce_fx                         | Lightning DDP                       | ❌                             | ❌                                    |

---

## 模块化程度

- **TorchMetrics**:      ████████████  最强 (functional/OOP/wrapper三层，可独立使用任一层)
- **lm-eval-harness**:   ██████████    强 (Registry/Task/Model/Filter/Evaluator 完全解耦)
- **cell-eval**:         ████████      良 (Registry/Pipeline/Evaluator分离，但 Profile 硬编码)
- **PerturBench**:       █████         中 (训练-评估耦合，Evaluation对象内部还行)
- **TDC**:               ████          中下 (metadata.py巨型单文件，BenchmarkGroup 思路好但实现粗)

---

## 可借鉴的设计模式 → 统一框架

**从每个项目偷什么？**

| 借鉴来源         | 模式/思路                                  | 怎么用/细节                                 |
|------------------|--------------------------------------------|---------------------------------------------|
| cell-eval        | Registry+Protocol metric 接口              | metric 注册不强制继承，函数或类都行            |
| cell-eval        | 不可变 dataclass                           | PerturbationData frozen dataclass 数据流通     |
| PerturBench      | 12 种聚合策略 (cell→pert)                  | 聚合作为独立可配置步骤，不硬编码               |
| PerturBench      | Pairwise + Ranking 评估                    | 不只看 per-pert 分数，还看模型排序扰动正确性   |
| PerturBench      | 多模型 dict 对比                           | 评估器原生支持 N 个模型同时比较                |
| lm-eval-harness  | YAML 定义 benchmark task                   | 新 benchmark = 新 YAML，不改代码               |
| lm-eval-harness  | TaskManager 自动发现                       | 扫描 benchmarks/ 目录自动注册                  |
| lm-eval-harness  | Filter pipeline                            | 预测后处理链（按 cell type 过滤、按 DE genes mask) |
| TorchMetrics     | update/compute/reset 流式模式              | 大数据集不用全部加载到内存                     |
| TorchMetrics     | MetricCollection 自动 compute              | 共享状态 metrics 只算一次（如 MAE/MSE/R²)      |
| TorchMetrics     | Wrapper 组合模式                           | BootStrapper 置信区间、ClasswiseWrapper 按 cell type 拆分 |
| TDC              | BenchmarkGroup 打包                        | benchmark = 数据集 + 指定 metrics + 固定 split |
| TDC              | evaluate_many (多 seed 聚合)               | 生物任务必须报 mean±std                        |

---

## 框架骨架草图

```
perteval/
├── metrics/              # ← cell-eval: Registry + Protocol，函数/类都能注册，惰性加载
│   ├── registry.py       # Metric 注册
│   ├── base.py           # Metric Protocol (update/compute，可选流式)
│   ├── functional/       # ← TorchMetrics: 纯函数计算核心
│   ├── wrappers/         # ← TorchMetrics: Bootstrap, CelltypeWise, DEGMask
│   └── collection.py     # ← TorchMetrics: MetricCollection，自动分组
├── aggregation/          # ← PerturBench: cell→pert 聚合策略
│   ├── average.py, logfc.py, pca.py, de_scores.py ...
├── benchmarks/           # ← TDC + lm-eval: YAML 定义 + 自动发现
│   ├── manager.py        # 扫描目录，加载 YAML
│   ├── group.py          # BenchmarkGroup(dataset+metrics+split+aggregation)
│   └── configs/*.yaml    # 每个 benchmark 一个 YAML
├── data/                 # ← cell-eval: 不可变 dataclass
│   ├── types.py          # PerturbationData, DEComparison (frozen)
│   └── loaders/          # ← TDC: accessor pattern
├── evaluation/           # ← lm-eval: Evaluator 编排
│   ├── evaluator.py      # 多模型 × 多 benchmark 矩阵评估
│   ├── pairwise.py       # ← PerturBench: N×N + ranking
│   └── filters/          # ← lm-eval: 后处理 pipeline
└── results/
    ├── output.py         # per-pert + agg DataFrame
    └── compare.py        # 多模型对比表 + evaluate_many (mean±std)
```

---

要进入正式的框架设计阶段吗？我可以把这个分析写进 docs，然后开始出详细设计文档。

---

✻ Worked for 2m 7s

❯ 来吧！superpower！

⏺ Skill(superpowers:brainstorming)
  ⎿  Successfully loaded skill

⏺ 我们在这个对话里已经做了大量的项目探索——对 7 个参考框架进行了架构分析、代码质量审计、功能模块对比，并且提炼了可借鉴的设计模式。让我快速确认一下当前状态。

Read 1 file (ctrl+o to expand)

⏺ Bash(git log --oneline -5)
  ⎿  c6cc315 Initial commit: benchmark frameworks comparison report

⏺ 好，项目上下文已经很充分了。进入澄清问题阶段。

我们在讨论中已经有了很多共识，但有一个关键问题需要先明确：

> 这个框架的核心用户是谁？
>
> - (A) 你自己 / 你的团队——用来统一评估你们开发的扰动预测模型，快速迭代
> - (B) 社区开源工具——像 lm-eval-harness 那样，让任何人都能用 pip install + 一行命令跑 benchmark
> - (C) 两者兼顾——先满足自用，但从一开始就按开源标准设计（文档、API、包发布）

这会直接影响 API 设计的优先级、配置复杂度、以及文档投入。
