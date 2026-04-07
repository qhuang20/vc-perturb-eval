"""Shared test fixtures for perteval test suite."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


def build_random_anndata(
    n_obs: int = 100,
    n_vars: int = 50,
    perturbations: list[str] | None = None,
    control_label: str = "control",
    perturbation_key: str = "perturbation",
    cell_type_key: str = "cell_type",
    cell_types: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> ad.AnnData:
    """Build a synthetic AnnData for testing.

    Returns an AnnData with:
    - X: random expression matrix (n_obs × n_vars), log-normalized scale
    - obs[perturbation_key]: perturbation labels
    - obs[cell_type_key]: cell type labels
    - var_names: gene_0, gene_1, ...
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if perturbations is None:
        perturbations = ["pertA", "pertB", "pertC"]
    if cell_types is None:
        cell_types = ["TypeA", "TypeB"]

    all_labels = [control_label] + perturbations
    obs_labels = rng.choice(all_labels, size=n_obs)
    obs_cell_types = rng.choice(cell_types, size=n_obs)

    X = rng.standard_normal((n_obs, n_vars)).astype(np.float32)
    # Shift expression for each perturbation to create detectable signal
    for i, pert in enumerate(perturbations, start=1):
        mask = obs_labels == pert
        X[mask] += i * 0.5

    adata = ad.AnnData(
        X=X,
        obs={
            perturbation_key: obs_labels,
            cell_type_key: obs_cell_types,
        },
    )
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    return adata


@pytest.fixture
def synthetic_adata(rng):
    """A synthetic AnnData with 3 perturbations + control."""
    return build_random_anndata(rng=rng)


@pytest.fixture
def synthetic_pair(rng):
    """A pair of synthetic AnnData (predicted, ground_truth) with matching structure."""
    ground_truth = build_random_anndata(n_obs=200, rng=rng)
    pred_rng = np.random.default_rng(99)
    predicted = ground_truth.copy()
    predicted.X = ground_truth.X + pred_rng.standard_normal(ground_truth.X.shape).astype(
        np.float32
    ) * 0.1
    return predicted, ground_truth
