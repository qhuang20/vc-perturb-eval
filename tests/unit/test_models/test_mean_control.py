import numpy as np

from tests.conftest import build_random_anndata


def test_mean_control_predict_shape():
    from perteval.models.baselines.mean_control import MeanControl

    rng = np.random.default_rng(42)
    adata = build_random_anndata(n_obs=200, perturbations=["pertA", "pertB"], rng=rng)
    model = MeanControl()
    model.train(adata, perturbation_key="perturbation", control_key="control")
    control_cells = adata[adata.obs["perturbation"] == "control"]
    predicted = model.predict(control_cells, perturbations=["pertA", "pertB"])
    assert predicted.n_obs > 0
    assert predicted.n_vars == adata.n_vars
    assert "perturbation" in predicted.obs.columns


def test_mean_control_predict_is_control_mean():
    from perteval.models.baselines.mean_control import MeanControl

    rng = np.random.default_rng(42)
    adata = build_random_anndata(n_obs=500, perturbations=["pertA"], rng=rng)
    model = MeanControl()
    model.train(adata, perturbation_key="perturbation", control_key="control")
    control_cells = adata[adata.obs["perturbation"] == "control"]
    predicted = model.predict(control_cells, perturbations=["pertA"])
    expected_mean = np.asarray(control_cells.X).mean(axis=0)
    pred_values = np.asarray(predicted[predicted.obs["perturbation"] == "pertA"].X)
    for i in range(pred_values.shape[0]):
        np.testing.assert_array_almost_equal(pred_values[i], expected_mean)


def test_mean_control_name():
    from perteval.models.baselines.mean_control import MeanControl

    model = MeanControl()
    assert model.name == "mean_control"


def test_model_registry_has_mean_control():
    from perteval.models.registry import model_registry

    assert "mean_control" in model_registry.list_available()
