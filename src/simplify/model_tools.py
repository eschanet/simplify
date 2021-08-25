import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import awkward1 as ak
import numpy as np
import pyhf


log = logging.getLogger(__name__)


def model_and_data(
    spec: Union[Dict[str, Any], pyhf.Workspace],
    poi_name: Optional[str] = None,
    asimov: bool = False,
    with_aux: bool = True,
) -> Tuple[pyhf.pdf.Model, List[float]]:
    """Returns model and data for a pyhf workspace spec in str or workspace format.

    Args:
        spec (Union[Dict[str, Any], pyhf.Workspace]): a pyhf workspace specification
        poi_name (Optional[str], optional): name of the POI. Defaults to None.
        asimov (bool, optional): whether to return Asimov data instead. Defaults to False.
        with_aux (bool, optional): whether auxiliary data should be returned. Defaults to True.

    Returns:
        Tuple[pyhf.pdf.Model, List[float]]:
            - HistFactory model in pyhf format
            - data associated to the likelihood
    """

    workspace = pyhf.Workspace(spec) if isinstance(spec, dict) else spec
    model = workspace.model(
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },  # polynomial interpolation and exponential extrapolation
        poi_name=poi_name,
    )
    if not asimov:
        data = workspace.data(model, with_aux=with_aux)
    else:
        data = get_asimov_data(model, with_aux=with_aux)
    return model, data


def get_parameter_names(model: pyhf.pdf.Model) -> List[str]:
    """Gets the pretty labels of all fit parameters. Expands gammas using indices.

    Args:
        model (pyhf.pdf.Model): HistFactory model in pyhf format

    Returns:
        List[str]: list of pretty names for fit params
    """

    param_labels = []
    for param in model.config.par_order:
        for idx in range(model.config.param_set(param).n_parameters):
            param_labels.append(
                f"{param}[{idx}]"
                if model.config.param_set(param).n_parameters > 1
                else param
            )
    return param_labels


def get_parameter_types(model: pyhf.pdf.Model) -> List[str]:
    """Gets the types of all fit parameters.

    Possible types are constrained or unconstrained.

    Args:
        model (pyhf.pdf.Model): pyhf model

    Returns:
        List[str]: types of fit params
    """

    param_types = []
    for param in model.config.par_order:
        for _ in range(model.config.param_set(param).n_parameters):  # NOQA
            param_types.append(
                "constrained"
                if model.config.param_set(param).constrained
                else "unconstrained"
            )
    return param_types


def get_prefit_uncertainties(model: pyhf.pdf.Model) -> np.ndarray:
    """Gets the before-fit parameter uncertainties

    Args:
        model (pyhf.pdf.Model): pyhf model from which to extract the parameters

    Returns:
        np.ndarray: before-fit uncertainties
    """

    prefit_unc = []
    for parameter in model.config.par_order:
        if (
            model.config.param_set(parameter).constrained
            and not model.config.param_set(parameter).suggested_fixed
        ):
            # constrained and/or non-fixed parameter
            prefit_unc += model.config.param_set(parameter).width()
        else:
            if model.config.param_set(parameter).n_parameters == 1:
                # unconstrained normalisation factor or a fixed parameter
                # -> no uncertainty
                prefit_unc.append(0.0)
            else:
                # shapefactor
                prefit_unc += [0.0] * model.config.param_set(parameter).n_parameters
    return np.asarray(prefit_unc)


def get_asimov_data(model: pyhf.Model, with_aux: bool = True) -> List[float]:
    """Gets the asimov dataset for a model

    Args:
        model (pyhf.Model): the model for which to construct the asimov data
        with_aux (bool, optional): with or without auxdata. Defaults to True.

    Returns:
        List[float]: asimov dataset
    """

    asimov_data = model.expected_data(
        get_asimov_parameters(model), include_auxdata=with_aux
    ).tolist()
    return asimov_data


def get_asimov_parameters(model: pyhf.pdf.Model) -> np.ndarray:
    """Gets the list of asimov parameters.

    Args:
        model (pyhf.pdf.Model): model for which to get asimov parameters

    Returns:
        np.ndarray: asimov parameter values in the same order as ``model.config.par_order``
    """

    auxdata_params = [
        p
        for p in model.config.auxdata_order
        for _ in range(model.config.param_set(p).n_parameters)
    ]

    asimov_parameters = []
    for parameter in model.config.par_order:
        aux_indices = [i for i, par in enumerate(auxdata_params) if par == parameter]
        if aux_indices:
            # best-fit value from auxiliary data
            inits = [
                aux for i, aux in enumerate(model.config.auxdata) if i in aux_indices
            ]
        else:
            # suggested inits from workspace for normfactors
            inits = model.config.param_set(parameter).suggested_init
        asimov_parameters += inits

    return np.asarray(asimov_parameters)


def _get_channel_bounds_indices(model: pyhf.pdf.Model) -> List[int]:
    """Gets indices for splitting a list of observations into
    different channels

    Args:
        model (pyhf.pdf.Model): model for which to get the bounds

    Returns:
        List[int]: list of channel bound indices
    """

    # number of bins per channel
    bins_per_channel = [model.config.channel_nbins[ch] for ch in model.config.channels]
    # indices where a new channel starts
    channel_start = [sum(bins_per_channel[:i]) for i in range(1, len(bins_per_channel))]
    return channel_start


def calculate_std(
    model: pyhf.pdf.Model,
    parameters: np.ndarray,
    uncertainty: np.ndarray,
    corr_mat: np.ndarray,
) -> ak.highlevel.Array:
    """Computes the symmetrized yield std dev.

    Args:
        model (pyhf.pdf.Model): pyhf model for which to compute the std for all bins
        parameters (np.ndarray): central values for all parameters
        uncertainty (np.ndarray): uncertainties for all parameters
        corr_mat (np.ndarray): correlations between parameters

    Returns:
        ak.highlevel.Array: awkward array containing channel arrays. Each array
        contains the stds for each bin in the respective channel
    """

    # indices at which regions start so we know
    # where to start and end new regions
    region_split_indices = _get_channel_bounds_indices(model)

    # these lists will contain the distributions with all params varied
    # within +-1 sigma uncertainty
    up_variations = []
    down_variations = []

    # compute model distribution for every parameter variation
    for i_par in range(model.config.npars):
        # one param varied within uncertainty
        # other params remain constant
        up_pars = parameters.copy()
        up_pars[i_par] += uncertainty[i_par]
        down_pars = parameters.copy()
        down_pars[i_par] -= uncertainty[i_par]

        # model distribution with one parameter varied up
        up_combined = model.expected_data(up_pars, include_auxdata=False)
        up_yields = np.split(up_combined, region_split_indices)
        up_variations.append(up_yields)

        # model distribution with one parameter varied down
        down_combined = model.expected_data(down_pars, include_auxdata=False)
        down_yields = np.split(down_combined, region_split_indices)
        down_variations.append(down_yields)

    # convert into awkward arrays
    up_variations = ak.from_iter(up_variations)
    down_variations = ak.from_iter(down_variations)

    # total variance with indices being channels and bins
    total_variance = ak.from_iter(
        [
            np.zeros(shape=(model.config.channel_nbins[ch]))
            for ch in model.config.channels
        ]
    )

    # loop over parameters to sum up total variance
    # start with diagonal of correlation matrix
    for i_par in range(model.config.npars):
        # add square of symmetric uncertainty to variance
        total_variance = (
            total_variance + ((up_variations[i_par] - down_variations[i_par]) / 2) ** 2
        )

    # on to the off-diagonal elements of the correlation matrix
    labels = get_parameter_names(model)
    if np.count_nonzero(corr_mat - np.diag(np.ones_like(parameters))) > 0:
        for i_par in range(model.config.npars):
            for j_par in range(model.config.npars):
                # only need to loop over one half of the matrix (it's symmetric)
                # we'll add a factor 2 below to consider the second half
                if j_par >= i_par:
                    continue

                # we could go for an approximate calculation here
                # e.g. only picking elements in the correlation matrix greater than 1e-05
                if (
                    labels[i_par][0:10] == "staterror_"
                    and labels[j_par][0:10] == "staterror_"
                ):
                    # statistical uncertainties are pair-wise orthogonal
                    continue
                sym_unc_i = (up_variations[i_par] - down_variations[i_par]) / 2
                sym_unc_j = (up_variations[j_par] - down_variations[j_par]) / 2
                total_variance = total_variance + 2 * (
                    corr_mat[i_par, j_par] * sym_unc_i * sym_unc_j
                )

    # return standard deviation
    log.debug(f"total std is {np.sqrt(total_variance)}")
    return np.sqrt(total_variance)
