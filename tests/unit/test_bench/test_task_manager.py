import pytest


def test_task_manager_load_yaml(tmp_path):
    from perteval.bench.task_manager import TaskManager

    yaml_content = """
dataset: norman19
metrics: [pearson_delta, mse]
split:
  method: random
  frac: [0.8, 0.1, 0.1]
  seed: 42
aggregation: average
"""
    (tmp_path / "test_bench.yaml").write_text(yaml_content)
    manager = TaskManager(str(tmp_path))
    config = manager.get("test_bench")
    assert config.dataset == "norman19"
    assert config.metrics == ["pearson_delta", "mse"]
    assert config.split_method == "random"
    assert config.aggregation == "average"


def test_task_manager_discover(tmp_path):
    from perteval.bench.task_manager import TaskManager

    (tmp_path / "bench_a.yaml").write_text("dataset: a\nmetrics: [mse]")
    (tmp_path / "bench_b.yaml").write_text("dataset: b\nmetrics: [mse]")
    (tmp_path / "not_yaml.txt").write_text("ignored")
    manager = TaskManager(str(tmp_path))
    available = manager.list_available()
    assert set(available) == {"bench_a", "bench_b"}


def test_task_manager_defaults(tmp_path):
    from perteval.bench.task_manager import TaskManager

    yaml_content = "dataset: norman19\nmetrics: [mse]\n"
    (tmp_path / "minimal.yaml").write_text(yaml_content)
    manager = TaskManager(str(tmp_path))
    config = manager.get("minimal")
    assert config.split_method == "random"
    assert config.split_frac == (0.8, 0.1, 0.1)
    assert config.aggregation == "average"
    assert config.split_seed == 42


def test_task_manager_missing_raises(tmp_path):
    from perteval.bench.task_manager import TaskManager

    manager = TaskManager(str(tmp_path))
    with pytest.raises(KeyError, match="not found"):
        manager.get("nonexistent")
