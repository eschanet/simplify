from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import iminuit
import numpy as np
import pyhf

from . import model_utils

import logging
from .logger import logger

log = logger.getChild(__name__)


class FitResults(NamedTuple):
    """
    Collects fit results in a single object.
    """

    bestfit: np.ndarray
    uncertainty: np.ndarray
    labels: List[str]
    corr_mat: np.ndarray
    best_twice_nll: float


def print_results(
    fit_result: FitResults,
) -> None:
    """
    Prints best-fit parameter results and uncertainties.
    """
    max_label_length = max([len(label) for label in fit_result.labels])
    for i, label in enumerate(fit_result.labels):
        log.info(
            f"{label.ljust(max_label_length)}: {fit_result.bestfit[i]: .6f} +/- "
            f"{fit_result.uncertainty[i]:.6f}"
        )


def _fit_model_pyhf(model: pyhf.pdf.Model, data: List[float]) -> FitResults:
    """
    Uses pyhf.infer API to perform a maximum likelihood fit.
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    result, result_obj = pyhf.infer.mle.fit(
        data, model, return_uncertainties=True, return_result_obj=True
    )

    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = model_utils.get_parameter_names(model)
    corr_mat = result_obj.minuit.np_matrix(correlation=True, skip_fixed=False)
    best_twice_nll = float(result_obj.fun)

    fit_result = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)
    return fit_result


def fit(spec: Dict[str, Any], asimov: bool = False) -> FitResults:
    """
    Performs a  maximum likelihood fit, reports and returns the results.
    The asimov flag allows to fit the Asimov dataset instead of observed
    data.
    """
    log.info("performing maximum likelihood fit")

    model, data = model_utils.model_and_data(spec, asimov=asimov)

    fit_result = _fit_model_pyhf(model, data)

    print_results(fit_result)
    log.debug(f"-2 log(L) = {fit_result.best_twice_nll:.6f} at the best-fit point")

    return fit_result
