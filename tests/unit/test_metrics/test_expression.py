import numpy as np
import pytest


def test_pearson_delta_perfect_prediction():
    from perteval.metrics.functional.expression import pearson_delta
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pearson_delta(pred, truth) == pytest.approx(1.0)


def test_pearson_delta_anticorrelated():
    from perteval.metrics.functional.expression import pearson_delta
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    truth = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert pearson_delta(pred, truth) == pytest.approx(-1.0)


def test_pearson_delta_constant_returns_zero():
    from perteval.metrics.functional.expression import pearson_delta
    pred = np.array([3.0, 3.0, 3.0])
    truth = np.array([1.0, 2.0, 3.0])
    assert pearson_delta(pred, truth) == pytest.approx(0.0)


def test_mse_zero_error():
    from perteval.metrics.functional.expression import mse
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.0, 2.0, 3.0])
    assert mse(pred, truth) == pytest.approx(0.0)


def test_mse_known_value():
    from perteval.metrics.functional.expression import mse
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([4.0, 5.0, 6.0])
    assert mse(pred, truth) == pytest.approx(9.0)


def test_mae_zero_error():
    from perteval.metrics.functional.expression import mae
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.0, 2.0, 3.0])
    assert mae(pred, truth) == pytest.approx(0.0)


def test_mae_known_value():
    from perteval.metrics.functional.expression import mae
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([4.0, 5.0, 6.0])
    assert mae(pred, truth) == pytest.approx(3.0)
