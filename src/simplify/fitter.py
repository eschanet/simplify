import logging
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pyhf

from . import model_tools


log = logging.getLogger(__name__)


class FitResults(NamedTuple):
    """Collects fit results in a single object."""

    bestfit: np.ndarray
    uncertainty: np.ndarray
    labels: List
    types: List
    corr_mat: np.ndarray
    best_twice_nll: float


def print_results(
    fit_result: FitResults,
) -> None:
    """Prints best-fit parameter results and uncertainties.

    Parameters
    ----------
    fit_result : FitResults
        Results of the fit to be printed.
    """
    max_label_length = max([len(label) for label in fit_result.labels])
    for i, label in enumerate(fit_result.labels):
        log.info(
            f"{label.ljust(max_label_length)}: {fit_result.bestfit[i]: .6f} +/- "
            f"{fit_result.uncertainty[i]:.6f}"
        )


def _fit_model_pyhf(
    model: pyhf.pdf.Model,
    data: List[float],
    init_pars: Optional[List[float]] = None,
    fixed_pars: Optional[List[bool]] = None,
) -> FitResults:
    """Uses pyhf.infer API to perform a maximum likelihood fit.

    Parameters
    ----------
    model : pyhf.pdf.Model
        Model to be used in the fit.
    data : List[float]
        Data to fit the model to.
    init_pars : Optional[List[float]]
        Initial parameter settings. Setting to none uses pyhf suggested ones.
    fixed_pars : Optional[List[float]]
        List of parameters to set to be fixed in the fit.

    Returns
    -------
    FitResults
        Results of the fit.

    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=False))

    result, result_obj = pyhf.infer.mle.fit(
        data,
        model,
        init_pars=init_pars,
        fixed_params=fixed_pars,
        return_uncertainties=True,
        return_result_obj=True,
    )

    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = model_tools.get_parameter_names(model)
    types = model_tools.get_parameter_types(model)
    corr_mat = result_obj.minuit.np_matrix(correlation=True, skip_fixed=False)
    best_twice_nll = float(result_obj.fun)

    # ordering things
    # _order = np.argsort(labels)
    # bestfit = bestfit[_order]
    # uncertainty = uncertainty[_order]
    # labels = labels[_order]
    # types = types[_order]
    # # corr_mat = corr_mat[_order]

    fit_result = FitResults(
        bestfit, uncertainty, labels, types, corr_mat, best_twice_nll
    )
    return fit_result


def fit(spec: Dict[str, Any], asimov: bool = False) -> FitResults:
    """Performs a  maximum likelihood fit, reports and returns the results.
    The asimov flag allows to fit the Asimov dataset instead of observed
    data.

    Parameters
    ----------
    spec : Dict[str, Any]
        JSON spec readable by `pyhf`.
    asimov : bool
        boolean that determines if the fit is done to
        asimov data or not. Defaults to fals.

    Returns
    -------
    FitResults
        Object containing fit results.
    """

    log.info("performing maximum likelihood fit")

    model, data = model_tools.model_and_data(spec, asimov=asimov)

    fit_result = _fit_model_pyhf(model, data)  # only one for now

    log.debug(print_results(fit_result))  # type: ignore
    log.debug(f"-2 log(L) = {fit_result.best_twice_nll:.6f} at the best-fit point")

    return fit_result
