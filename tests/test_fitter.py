import logging
from unittest import mock

import numpy as np
import pytest

from simplify import fitter
from simplify import model_tools


def test_FitResults():
    bestfit = np.asarray([1.0])
    uncertainty = np.asarray([0.1])
    labels = ["param"]
    types = ["constrained"]
    corr_mat = np.asarray([[1.0]])
    best_twice_nll = 2.0
    fit_results = fitter.FitResults(
        bestfit, uncertainty, labels, types, corr_mat, best_twice_nll
    )
    assert np.allclose(fit_results.bestfit, bestfit)
    assert np.allclose(fit_results.uncertainty, uncertainty)
    assert fit_results.labels == labels
    assert fit_results.types == types
    assert np.allclose(fit_results.corr_mat, corr_mat)
    assert fit_results.best_twice_nll == best_twice_nll


def test_print_results(caplog):
    caplog.set_level(logging.DEBUG)

    bestfit = np.asarray([1.0, 2.0])
    uncertainty = np.asarray([0.1, 0.3])
    labels = ["param_1", "param_2"]
    types = ["constained", "constrained"]
    fit_results = fitter.FitResults(  # NOQA
        bestfit, uncertainty, labels, types, np.empty(0), 0.0
    )  # NOQA

    fitter.print_results(fit_results)
    assert "param_1:  1.000000 +/- 0.100000" in [rec.message for rec in caplog.records]
    assert "param_2:  2.000000 +/- 0.300000" in [rec.message for rec in caplog.records]
    caplog.clear()


# skip a "RuntimeWarning: numpy.ufunc size changed" warning
# due to different numpy versions used in dependencies
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test__fit_model_pyhf(example_spec, example_spec_multibin):
    model, data = model_tools.model_and_data(example_spec)
    fit_results = fitter._fit_model_pyhf(model, data)
    assert np.allclose(fit_results.bestfit, [1.1, 5.58731303])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.21248646])
    assert fit_results.labels == ["staterror_SR", "mu_Sig"]
    assert np.allclose(fit_results.best_twice_nll, 6.850287450660111)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # # TODO: Asimov fit, with fixed gamma (fixed not to Asimov MLE)
    # model, data = model_tools.model_and_data(example_spec, asimov=True)
    # fit_results = fitter._fit_model_pyhf(model, data)
    # # the gamma factor is multiplicative and fixed to 1.1, so the
    # # signal strength needs to be 1/1.1 to compensate
    # assert np.allclose(fit_results.bestfit, [1.1, 0.90917877])
    # assert np.allclose(fit_results.uncertainty, [0.0, 0.12623179])
    # assert fit_results.labels == ["staterror_SR", "mu_Sig"]
    # assert np.allclose(fit_results.best_twice_nll, 5.68851093)
    # assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # parameters held constant via keyword argument
    model, data = model_tools.model_and_data(example_spec_multibin)
    init_pars = model.config.suggested_init()
    init_pars[0] = 0.9
    init_pars[1] = 1.1
    fixed_pars = model.config.suggested_fixed()
    fixed_pars[0] = True
    fixed_pars[1] = True
    fit_results = fitter._fit_model_pyhf(
        model, data, init_pars=init_pars, fixed_pars=fixed_pars
    )
    assert np.allclose(fit_results.bestfit, [0.9, 1.1, 1.11996446, 0.96618774])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.0, 0.1476617, 0.17227148])
    assert np.allclose(fit_results.best_twice_nll, 11.2732492)


@mock.patch("simplify.fitter.print_results")
@mock.patch(
    "simplify.fitter._fit_model_pyhf",
    return_value=fitter.FitResults(
        np.asarray([1.0]), np.asarray([0.1]), ["par"], ["constrained"], np.empty(0), 2.0
    ),
)
@mock.patch("simplify.model_tools.model_and_data", return_value=("model", "data"))
def test_fit(mock_load, mock_fitter, mock_print, example_spec):
    # fit through pyhf.infer API
    fit_results = fitter.fit(example_spec)
    assert mock_load.call_args_list == [[(example_spec,), {"asimov": False}]]
    assert mock_fitter.call_args_list == [[("model", "data")]]
    mock_print.assert_called_once()
    assert mock_print.call_args[0][0].bestfit == [1.0]
    assert mock_print.call_args[0][0].uncertainty == [0.1]
    assert mock_print.call_args[0][0].labels == ["par"]
    assert mock_print.call_args[0][0].types == ["constrained"]
    assert fit_results.bestfit == [1.0]

    # # TODO: Asimov fit
    # fit_results = fitter.fit(example_spec, asimov=True)
    # assert mock_fitter.call_count == 2
    # assert mock_load.call_args == [(example_spec,), {"asimov": True}]
    # assert fit_results.bestfit == [1.0]
