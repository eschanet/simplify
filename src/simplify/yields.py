import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import awkward1 as ak
import numpy as np

from . import fitter
from . import model_tools


log = logging.getLogger(__name__)


class Yields(NamedTuple):
    """Collects yields in a single object"""

    regions: List[str]
    samples: List[str]
    yields: Dict[str, np.ndarray]
    uncertainties: Dict[str, ak.highlevel.Array]
    data: Dict[str, np.ndarray]


def _pdgRound(
    value: float,
    error: float,
) -> Tuple[str, str]:
    """
    Given a value and an error, round and format them according to PDG rounding rules.
    """

    def threeDigits(err: float) -> int:
        """Extract the three most significant digits and return as int"""
        return int(
            ("%.2e" % float(err))
            .split('e')[0]
            .replace('.', '')
            .replace('+', '')
            .replace('-', '')
        )

    def nSignificantDigits(threeDigits: int) -> int:
        if threeDigits == 0:
            return 0
        assert threeDigits < 1000, (
            "three digits (%d) cannot be larger than 10^3" % threeDigits
        )
        assert threeDigits >= 100, (
            "three digits (%d) cannot be smaller than 10^2" % threeDigits
        )
        if threeDigits < 355:
            return 2
        elif threeDigits < 950:
            return 1
        else:
            return 2

    def frexp10(value: float) -> Tuple[float, int]:
        "convert to mantissa+exp representation (same as frex, but in base 10)"
        valueStr = ("%e" % float(value)).split('e')
        return float(valueStr[0]), int(valueStr[1])

    def nDigitsValue(expVal: int, expErr: int, nDigitsErr: int) -> int:
        """
        compute the number of digits we want for the value,
        assuming we keep nDigitsErr for the error
        """
        return expVal - expErr + nDigitsErr

    def formatValue(
        value: float, exponent: int, nDigits: int, extraRound: int = 0
    ) -> str:
        "Format the value; extraRound is meant for the special case of threeDigits>950"
        roundAt = nDigits - 1 - exponent - extraRound
        nDec = roundAt if exponent < nDigits else 0
        nDec = max([nDec, 0])
        return ('%.' + str(nDec) + 'f') % round(value, roundAt)

    if value == 0.0 and error == 0.0:
        return ("0.0", "0.0")
    tD = threeDigits(error)
    nD = nSignificantDigits(tD)
    expVal, expErr = frexp10(value)[1], frexp10(error)[1]
    extraRound = 1 if tD >= 950 else 0
    return (
        formatValue(value, expVal, nDigitsValue(expVal, expErr, nD), extraRound),
        formatValue(error, expErr, nD, extraRound),
    )


def _get_data_yield_uncertainties(
    spec: Dict[str, Any],
    fit_results: Optional[fitter.FitResults] = None,
    exclude_process: Optional[List[str]] = None,
) -> Yields:
    """Gets data, yields and uncertainties.
    Prefit if no fit results are given, else postfit.

    Parameters
    ----------
    spec : Dict[str, Any]
        pyhf JSON spec.
    fit_results : Optional[fitter.FitResults]
        Fit results holding parameters and best-fit values.
    exclude_process : Optional[List[str]]
        List of sample names to exclude from yields.

    Returns
    -------
    Yields
        Data, yields and uncertainties.

    """

    model, data_combined = model_tools.model_and_data(spec, with_aux=False)

    if fit_results is not None:
        param_values = fit_results.bestfit
        param_uncertainty = fit_results.uncertainty
        corr_mat = fit_results.corr_mat

    else:
        # no fit results specified, draw a prefit plot
        param_values = model_tools.get_asimov_parameters(model)
        param_uncertainty = model_tools.get_prefit_uncertainties(model)
        corr_mat = np.zeros(shape=(len(param_values), len(param_values)))
        np.fill_diagonal(corr_mat, 1.0)

    yields_combined = model.main_model.expected_data(
        param_values, return_by_sample=True
    )  # all channels concatenated

    # Need to slice the yields into an array where
    # first index is the channel and second index is the sample
    region_split_indices = model_tools._get_channel_boundary_indices(model)
    model_yields = np.split(yields_combined, region_split_indices, axis=1)
    data_vals = np.split(
        data_combined, region_split_indices
    )  # data just indexed by channel

    # calculate the total standard deviation of the model prediction, index: channel
    total_stdev_model = model_tools.calculate_stdev(
        model, param_values, param_uncertainty, corr_mat
    )

    exclude_process = exclude_process or []
    include_samples = np.array(
        [
            True if sample not in exclude_process else False
            for sample in model.config.samples
        ]
    )

    yields = {
        channel: model_yields[i_yields][include_samples]
        for i_yields, channel in enumerate(model.config.channels)
    }
    uncertainties = {
        channel: total_stdev_model[i_unc]
        for i_unc, channel in enumerate(model.config.channels)
    }
    data = {
        channel: data_vals[i_data]
        for i_data, channel in enumerate(model.config.channels)
    }

    return Yields(
        model.config.channels,
        list(np.array(model.config.samples)[include_samples]),
        yields,
        uncertainties,
        data,
    )


def get_yields(
    spec: Dict[str, Any],
    fit_results: Optional[fitter.FitResults] = None,
    exclude_process: Optional[List[str]] = None,
) -> Yields:
    """Gets yields and uncertainties. Prefit if no fit results are given, else postfit.

    Parameters
    ----------
    spec : Dict[str, Any]
        pyhf JSON spec.
    fit_results : Optional[fitter.FitResults]
        Fit results holding parameters and best-fit values.
    exclude_process : Optional[List[str]]
        List of sample names to exclude from yields.

    Returns
    -------
    Yields
        Yields and uncertainties for all channels.

    """

    return _get_data_yield_uncertainties(
        spec, fit_results=fit_results, exclude_process=exclude_process
    )
