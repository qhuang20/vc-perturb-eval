import pytest

from tests.conftest import build_random_anndata


def test_local_accessor_load(tmp_path):
    from perteval.data.accessors.local import LocalAccessor

    adata = build_random_anndata(n_obs=50)
    path = tmp_path / "test.h5ad"
    adata.write_h5ad(path)
    accessor = LocalAccessor(str(tmp_path))
    loaded = accessor.load("test")
    assert loaded.n_obs == 50
    assert loaded.n_vars == 50


def test_local_accessor_list_datasets(tmp_path):
    from perteval.data.accessors.local import LocalAccessor

    adata = build_random_anndata(n_obs=10)
    (tmp_path / "datasetA.h5ad").touch()
    adata.write_h5ad(tmp_path / "datasetB.h5ad")
    accessor = LocalAccessor(str(tmp_path))
    datasets = accessor.list_datasets()
    assert "datasetA" in datasets
    assert "datasetB" in datasets


def test_local_accessor_missing_file_raises(tmp_path):
    from perteval.data.accessors.local import LocalAccessor

    accessor = LocalAccessor(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not found"):
        accessor.load("nonexistent")
