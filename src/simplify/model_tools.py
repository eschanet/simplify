import pyhf
import numpy as np

from typing import Any, Dict, List, Tuple, Optional, overload

import awkward1 as ak

from . import fitter

import logging
log = logging.getLogger(__name__)


@overload
def model_and_data(
        spec: Dict[str, Any], poi_name: str = None, asimov: bool = False, with_aux: bool = True
) -> Tuple[pyhf.pdf.Model, List[float]]:
    ...


def model_and_data(
    spec: Dict[str, Any], poi_name: str = None, asimov: bool = False, with_aux: bool = True
) -> Tuple[pyhf.pdf.Model, List[float]]:
    """Returns model and data for a ``pyhf`` workspace spec.
    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to return the Asimov dataset, defaults
            to False
        with_aux (bool, optional): whether to also return auxdata, defaults
            to True
    Returns:
        Tuple[pyhf.pdf.Model, List[float]]:
            - a HistFactory-style model in ``pyhf`` format
            - the data (plus auxdata if requested) for the model
    """

    if isinstance(spec, dict):
        workspace = pyhf.Workspace(spec)
    elif isinstance(spec, pyhf.Workspace):
        workspace = spec

    model = workspace.model(
        modifier_settings = {
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
        poi_name = poi_name
    )
    if not asimov:
        data = workspace.data(model, with_aux = with_aux)
    else:
        data = build_Asimov_data(model, with_aux = with_aux)
    return model, data


def get_parameter_names(model: pyhf.pdf.Model) -> List[str]:
    """Gets the labels of all fit parameters. Expands gammas.

    Parameters
    ----------
    model : pyhf.pdf.Model
        pyhf model.

    Returns
    -------
    List[str]
        Names of fit parameters.

    """

    labels = []
    for param in model.config.par_order:
        for i_par in range(model.config.param_set(param).n_parameters):
            labels.append(
                f"{param}[{i_par}]"
                if model.config.param_set(param).n_parameters > 1
                else param
            )
    return labels

def get_parameter_types(model: pyhf.pdf.Model) -> List[str]:
    """Gets the types of all fit parameters. Expands gammas.

    Parameters
    ----------
    model : pyhf.pdf.Model
        pyhf model.

    Returns
    -------
    List[str]
        Types of fit parameters.

    """
    types = []
    for param in model.config.par_order:
        for i_par in range(model.config.param_set(param).n_parameters):
            types.append(
                f"constrained"
                if model.config.param_set(param).constrained
                else f"unconstrained"
            )
    return types


def get_prefit_uncertainties(model: pyhf.pdf.Model) -> np.ndarray:
    """Gets list of before fit parameter uncertainties for a pyhf model

    Parameters
    ----------
    model : pyhf.pdf.Model
        pyhf model for which the parameters should be extracted.

    Returns
    -------
    np.ndarray
        Before fit uncertainties.

    """

    pre_fit_unc = []
    for parameter in model.config.par_order:
        if (
            model.config.param_set(parameter).constrained
            and not model.config.param_set(parameter).suggested_fixed
        ):
            pre_fit_unc += model.config.param_set(parameter).width()
        else:
            if model.config.param_set(parameter).n_parameters == 1:
                pre_fit_unc.append(0.0)
            else: # shapefactor
                pre_fit_unc += [0.0] * model.config.param_set(parameter).n_parameters
    return np.asarray(pre_fit_unc)


def build_Asimov_data(model: pyhf.Model, with_aux: bool = True) -> List[float]:
    pass


def get_asimov_parameters(model: pyhf.pdf.Model) -> np.ndarray:
    pass


def _get_channel_boundary_indices(model: pyhf.pdf.Model) -> List[int]:
    """Gets indices for splitting a list of observations into channels."""

    # get the amount of bins per channel
    bins_per_channel = [model.config.channel_nbins[ch] for ch in model.config.channels]
    # indices of positions where a new channel starts (from the second channel onwards)
    channel_start = [sum(bins_per_channel[:i]) for i in range(1, len(bins_per_channel))]
    return channel_start


def calculate_stdev(
    model: pyhf.pdf.Model,
    parameters: np.ndarray,
    uncertainty: np.ndarray,
    corr_mat: np.ndarray,
) -> ak.highlevel.Array:
    """Method for computing the symmetrized yield standard deviation of a model.

    Parameters
    ----------
    model : pyhf.pdf.Model
        pyhf model for which to compute the stdev for all bins.
    parameters : np.ndarray
        central values.
    uncertainty : np.ndarray
        uncertainties of all parameters.
    corr_mat : np.ndarray
        correlation matrix.

    Returns
    -------
    ak.highlevel.Array
        array containing channels with each channel being an array containing the stdevs for each bin.

    """

    # indices where to split to separate all bins into regions
    region_split_indices = _get_channel_boundary_indices(model)

    # the lists up_variations and down_variations will contain the model distributions
    # with all parameters varied individually within uncertainties
    # indices: variation, channel, bin
    up_variations = []
    down_variations = []

    # calculate the model distribution for every parameter varied up and down
    # within the respective uncertainties
    for i_par in range(model.config.npars):
        # central parameter values, but one parameter varied within uncertainties
        up_pars = parameters.copy()
        up_pars[i_par] += uncertainty[i_par]
        down_pars = parameters.copy()
        down_pars[i_par] -= uncertainty[i_par]

        # total model distribution with this parameter varied up
        up_combined = model.expected_data(up_pars, include_auxdata=False)
        up_yields = np.split(up_combined, region_split_indices)
        up_variations.append(up_yields)

        # total model distribution with this parameter varied down
        down_combined = model.expected_data(down_pars, include_auxdata=False)
        down_yields = np.split(down_combined, region_split_indices)
        down_variations.append(down_yields)

    # convert to awkward arrays for further processing
    up_variations = ak.from_iter(up_variations)
    down_variations = ak.from_iter(down_variations)

    # total variance, indices are: channel, bin
    total_variance_list = [
        np.zeros(shape=(model.config.channel_nbins[ch])) for ch in model.config.channels
    ]  # list of arrays, each array has as many entries as there are bins
    total_variance = ak.from_iter(total_variance_list)

    # loop over parameters to sum up total variance
    # first do the diagonal of the correlation matrix
    for i_par in range(model.config.npars):
        symmetric_uncertainty = (up_variations[i_par] - down_variations[i_par]) / 2
        total_variance = total_variance + symmetric_uncertainty ** 2

    labels = get_parameter_names(model)
    # continue with off-diagonal contributions if there are any
    if np.count_nonzero(corr_mat - np.diag(np.ones_like(parameters))) > 0:
        # loop over pairs of parameters
        for i_par in range(model.config.npars):
            for j_par in range(model.config.npars):
                if j_par >= i_par:
                    continue  # only loop over the half the matrix due to symmetry
                corr = corr_mat[i_par, j_par]
                # an approximate calculation could be done here by requiring
                # e.g. abs(corr) > 1e-5 to continue
                if (
                    labels[i_par][0:10] == "staterror_"
                    and labels[j_par][0:10] == "staterror_"
                ):
                    continue  # two different staterrors are orthogonal, no contribution
                sym_unc_i = (up_variations[i_par] - down_variations[i_par]) / 2
                sym_unc_j = (up_variations[j_par] - down_variations[j_par]) / 2
                # factor of two below is there since loop is only over half the matrix
                total_variance = total_variance + 2 * (corr * sym_unc_i * sym_unc_j)

    # convert to standard deviation
    total_stdev = np.sqrt(total_variance)
    log.debug(f"total stdev is {total_stdev}")
    return total_stdev
