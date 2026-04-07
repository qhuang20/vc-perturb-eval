"""perteval CLI — command-line interface for running evaluations."""

from __future__ import annotations

import argparse
import sys

import perteval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="perteval",
        description="Unified evaluation framework for single-cell perturbation prediction",
    )
    parser.add_argument("--version", action="version", version=f"perteval {perteval.__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # --- perteval run ---
    run_parser = subparsers.add_parser("run", help="Run benchmark evaluation")
    run_parser.add_argument("--benchmark", nargs="+", required=True, help="Benchmark name(s)")
    run_parser.add_argument("--model", nargs="+", required=True, help="Model name(s)")
    run_parser.add_argument("--benchmarks-dir", default="benchmarks", help="Benchmarks YAML dir")
    run_parser.add_argument("--data-dir", default="data", help="Data directory")
    run_parser.add_argument("--output", "-o", default="results", help="Output path prefix")

    # --- perteval evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate prediction vs ground truth")
    eval_parser.add_argument("--predicted", required=True, help="Predicted h5ad path")
    eval_parser.add_argument("--ground-truth", required=True, help="Ground truth h5ad path")
    eval_parser.add_argument("--metrics", nargs="+", default=None, help="Metric names")
    eval_parser.add_argument("--output", "-o", default="results", help="Output path prefix")

    # --- perteval list ---
    list_parser = subparsers.add_parser("list", help="List available resources")
    list_parser.add_argument("resource", choices=["models", "metrics", "benchmarks"], help="Resource type")
    list_parser.add_argument("--benchmarks-dir", default="benchmarks")

    args = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "list":
        _cmd_list(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_run(args):
    from perteval.bench.runner import BenchmarkRunner
    runner = BenchmarkRunner(benchmarks=args.benchmark, models=args.model,
                             benchmarks_dir=args.benchmarks_dir, data_dir=args.data_dir)
    results = runner.run()
    for bench_name, model_results in results.items():
        for model_name, eval_result in model_results.items():
            prefix = f"{args.output}_{bench_name}_{model_name}"
            eval_result.to_json(f"{prefix}.json")
            eval_result.to_csv(prefix)
            print(f"Results saved: {prefix}.json")


def _cmd_evaluate(args):
    import anndata as ad
    from perteval.bench.evaluator import Evaluator
    from perteval.data.types import PerturbationData
    predicted = ad.read_h5ad(args.predicted)
    ground_truth = ad.read_h5ad(args.ground_truth)
    data = PerturbationData(predicted=predicted, ground_truth=ground_truth)
    evaluator = Evaluator()
    metrics = args.metrics if args.metrics else "default"
    result = evaluator.evaluate(data, metrics=metrics)
    result.to_json(f"{args.output}.json")
    result.to_csv(args.output)
    print(f"Results saved: {args.output}.json")
    print(result.aggregated)


def _cmd_list(args):
    match args.resource:
        case "models":
            from perteval.models.registry import model_registry
            for name in model_registry.list_available():
                print(f"  {name}")
        case "metrics":
            from perteval.metrics.registry import metric_registry
            for name in metric_registry.list_available():
                info = metric_registry.get(name)
                print(f"  {name} ({info.metric_type.value}) — {info.description}")
        case "benchmarks":
            from perteval.bench.task_manager import TaskManager
            manager = TaskManager(args.benchmarks_dir)
            for name in manager.list_available():
                print(f"  {name}")
