import numpy as np
import pytest


def test_edistance_identical_zero():
    from perteval.metrics.functional.distribution import edistance

    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    assert edistance(X, X) == pytest.approx(0.0, abs=1e-10)


def test_edistance_different_positive():
    from perteval.metrics.functional.distribution import edistance

    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    Y = rng.standard_normal((50, 10)) + 3.0
    result = edistance(X, Y)
    assert result > 0


def test_edistance_symmetric():
    from perteval.metrics.functional.distribution import edistance

    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5))
    Y = rng.standard_normal((30, 5)) + 1.0
    assert edistance(X, Y) == pytest.approx(edistance(Y, X))
