import pyhf
import numpy as np

from typing import Any, Dict, List, Tuple, Optional, NamedTuple, overload

import awkward1 as ak

from . import fitter
from . import model_utils

import logging
from . import logger
log = logging.getLogger(__name__)

class Yields(NamedTuple):
    """Collects yields in a single object"""

    regions: List[str]
    samples: List[str]
    yields: Dict[str, np.array]
    uncertainties: Dict[str, np.ndarray]
    data: Dict[str, np.array]

def _pdgRound(
    value: float,
    error: float,
) -> Tuple[float, float]:
    """
    Given a value and an error, round and format them according to PDG rounding rules.
    """
    def threeDigits(err):
        """Extract the three most significant digits and return as int"""
        return int(("%.2e"%float(err)).split('e')[0].replace('.','').replace('+','').replace('-',''))
    def nSignificantDigits(threeDigits):
        if threeDigits==0: return 0
        assert threeDigits<1000,"three digits (%d) cannot be larger than 10^3"%threeDigits
        assert threeDigits>=100,"three digits (%d) cannot be smaller than 10^2"%threeDigits
        if threeDigits<355 : return 2
        elif threeDigits<950 : return 1
        else : return 2
    def frexp10(value) :
        "convert to mantissa+exp representation (same as frex, but in base 10)"
        valueStr = ("%e"%float(value)).split('e')
        return float(valueStr[0]), int(valueStr[1])
    def nDigitsValue(expVal, expErr, nDigitsErr) :
        "compute the number of digits we want for the value, assuming we keep nDigitsErr for the error"
        return expVal-expErr+nDigitsErr
    def formatValue(value, exponent, nDigits, extraRound=0) :
        "Format the value; extraRound is meant for the special case of threeDigits>950"
        roundAt = nDigits-1-exponent - extraRound
        nDec = roundAt if exponent<nDigits else 0
        nDec = max([nDec, 0])
        return ('%.'+str(nDec)+'f')%round(value,roundAt)
    if value == 0.0 and error == 0.0:
        return (0.0,0.0)
    tD = threeDigits(error)
    nD = nSignificantDigits(tD)
    expVal, expErr = frexp10(value)[1], frexp10(error)[1]
    extraRound = 1 if tD>=950 else 0
    return (formatValue(value, expVal, nDigitsValue(expVal, expErr, nD), extraRound),
            formatValue(error,expErr, nD, extraRound))


def _get_data_yield_uncertainties(
    spec: Dict[str, Any],
    fit_results: Optional[fitter.FitResults] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], ak.highlevel.Array]:
    """Gets data, yields and uncertainties. Prefit if no fit results are given, else postfit.

    Parameters
    ----------
    spec : Dict[str, Any]
        pyhf JSON spec.
    fit_results : Optional[fitter.FitResults]
        Fit results holding parameters and best-fit values.

    Returns
    -------
    Tuple[List[np.ndarray], ak.highlevel.Array]
        Data, yields and uncertainties.

    """

    model, data_combined = model_utils.model_and_data(spec, with_aux=False)

    if fit_results is not None:
        prefit = False
        param_values = fit_results.bestfit
        param_uncertainty = fit_results.uncertainty
        corr_mat = fit_results.corr_mat

    # else:
    #     # no fit results specified, draw a pre-fit plot
    #     prefit = True
    #     # use pre-fit parameter values, uncertainties, and diagonal correlation matrix
    #     param_values = get_asimov_parameters(model)
    #     param_uncertainty = get_prefit_uncertainties(model)
    #     corr_mat = np.zeros(shape=(len(param_values), len(param_values)))
    #     np.fill_diagonal(corr_mat, 1.0)

    yields_combined = model.main_model.expected_data(
        param_values, return_by_sample=True
    )  # all channels concatenated

    # Need to slice the yields into an array where first index is the channel and second index is the sample
    region_split_indices = model_utils._get_channel_boundary_indices(model)
    model_yields = np.split(yields_combined, region_split_indices, axis=1)
    data_vals = np.split(data_combined, region_split_indices)  # data just indexed by channel

    # calculate the total standard deviation of the model prediction, index: channel
    total_stdev_model = model_utils.calculate_stdev(
        model, param_values, param_uncertainty, corr_mat
    )

    yields = {channel : model_yields[i_yields] for i_yields, channel in enumerate(model.config.channels)}
    uncertainties = {channel : total_stdev_model[i_unc] for i_unc, channel in enumerate(model.config.channels)}
    data = {channel : data_vals[i_data] for i_data, channel in enumerate(model.config.channels)}

    return Yields(model.config.channels, model.config.samples, yields, uncertainties, data)


def get_yields(
    spec: Dict[str, Any],
    fit_results: Optional[fitter.FitResults] = None,
) -> Tuple[List[np.ndarray], ak.highlevel.Array]:
    """Gets yields and uncertainties. Prefit if no fit results are given, else postfit.

    Parameters
    ----------
    spec : Dict[str, Any]
        pyhf JSON spec.
    fit_results : Optional[fitter.FitResults]
        Fit results holding parameters and best-fit values.

    Returns
    -------
    Tuple[List[np.ndarray], ak.highlevel.Array]
        Yields and uncertainties for all channels.

    """

    return _get_data_yield_uncertainties(spec, fit_results)
