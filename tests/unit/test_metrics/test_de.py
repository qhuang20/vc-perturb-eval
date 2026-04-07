import numpy as np
import pytest


def test_overlap_at_k_perfect():
    from perteval.metrics.functional.de import overlap_at_k
    pred_genes = np.array(["g1", "g2", "g3", "g4", "g5"])
    truth_genes = np.array(["g1", "g2", "g3", "g4", "g5"])
    assert overlap_at_k(pred_genes, truth_genes, k=5) == pytest.approx(1.0)


def test_overlap_at_k_no_overlap():
    from perteval.metrics.functional.de import overlap_at_k
    pred_genes = np.array(["a1", "a2", "a3", "a4", "a5"])
    truth_genes = np.array(["b1", "b2", "b3", "b4", "b5"])
    assert overlap_at_k(pred_genes, truth_genes, k=5) == pytest.approx(0.0)


def test_overlap_at_k_partial():
    from perteval.metrics.functional.de import overlap_at_k
    pred_genes = np.array(["g1", "g2", "g3", "a1", "a2"])
    truth_genes = np.array(["g1", "g2", "g3", "b1", "b2"])
    assert overlap_at_k(pred_genes, truth_genes, k=5) == pytest.approx(0.6)


def test_overlap_at_k_truncates_to_k():
    from perteval.metrics.functional.de import overlap_at_k
    pred_genes = np.array(["g1", "g2", "a1", "a2", "a3"])
    truth_genes = np.array(["g1", "g2", "b1", "b2", "b3"])
    assert overlap_at_k(pred_genes, truth_genes, k=2) == pytest.approx(1.0)
