import glob
import pathlib
import logging

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyhf

from . import model_utils
from . import fit

from .helpers import plotting

log = logging.getLogger(__name__)

# def data_MC(
#     config: Dict[str, Any],
#     figure_folder: Union[str, pathlib.Path],
#     spec: Dict[str, Any],
#     fit_results: Optional[fit.FitResults] = None,
#     log_scale: Optional[bool] = None,
#     method: str = "matplotlib",
# ) -> None:
#     """Draws pre- and post-fit data/MC histograms from a ``pyhf`` workspace.
#     Args:
#         config (Dict[str, Any]): cabinetry configuration
#         figure_folder (Union[str, pathlib.Path]): path to the folder to save figures in
#         spec (Dict[str, Any]): ``pyhf`` workspace specification
#         fit_results (Optional[fit.FitResults]): parameter configuration to use for plot,
#             includes best-fit settings and uncertainties, as well as correlation matrix,
#             defaults to None (then the pre-fit configuration is drawn)
#         log_scale (Optional[bool], optional): whether to use logarithmic vertical axis,
#             defaults to None (automatically determine whether to use linear/log scale)
#         method (str, optional): backend to use for plotting, defaults to "matplotlib"
#     Raises:
#         NotImplementedError: when trying to plot with a method that is not supported
#     """
#     model, data_combined = model_utils.model_and_data(spec, with_aux=False)
#
#     if fit_results is not None:
#         # fit results specified, draw a post-fit plot with them applied
#         prefit = False
#         param_values = fit_results.bestfit
#         param_uncertainty = fit_results.uncertainty
#         corr_mat = fit_results.corr_mat
#
#     else:
#         # no fit results specified, draw a pre-fit plot
#         prefit = True
#         # use pre-fit parameter values, uncertainties, and diagonal correlation matrix
#         param_values = model_utils.get_asimov_parameters(model)
#         param_uncertainty = model_utils.get_prefit_uncertainties(model)
#         corr_mat = np.zeros(shape=(len(param_values), len(param_values)))
#         np.fill_diagonal(corr_mat, 1.0)
#
#     yields_combined = model.main_model.expected_data(
#         param_values, return_by_sample=True
#     )  # all channels concatenated
#
#     # slice the yields into an array where the first index is the channel,
#     # and the second index is the sample
#     region_split_indices = model_utils._get_channel_boundary_indices(model)
#     model_yields = np.split(yields_combined, region_split_indices, axis=1)
#     data = np.split(data_combined, region_split_indices)  # data just indexed by channel
#
#     # calculate the total standard deviation of the model prediction, index: channel
#     total_stdev_model = model_utils.calculate_stdev(
#         model, param_values, param_uncertainty, corr_mat
#     )
#
#     for i_chan, channel_name in enumerate(
#         model.config.channels
#     ):  # process channel by channel
#         histogram_dict_list = []  # one dict per region/channel
#
#         # get the region dictionary from the config for binning / variable name
#         region_dict = configuration.get_region_dict(config, channel_name)
#         bin_edges = template_builder._get_binning(region_dict)
#         variable = region_dict["Variable"]
#
#         for i_sam, sample_name in enumerate(model.config.samples):
#             histogram_dict_list.append(
#                 {
#                     "label": sample_name,
#                     "isData": False,
#                     "yields": model_yields[i_chan][i_sam],
#                     "variable": variable,
#                 }
#             )
#
#         # add data sample
#         histogram_dict_list.append(
#             {
#                 "label": "Data",
#                 "isData": True,
#                 "yields": data[i_chan],
#                 "variable": variable,
#             }
#         )
#
#         if method == "matplotlib":
#             from .contrib import matplotlib_visualize
#
#             if prefit:
#                 figure_path = pathlib.Path(figure_folder) / _build_figure_name(
#                     channel_name, True
#                 )
#             else:
#                 figure_path = pathlib.Path(figure_folder) / _build_figure_name(
#                     channel_name, False
#                 )
#             matplotlib_visualize.data_MC(
#                 histogram_dict_list,
#                 np.asarray(total_stdev_model[i_chan]),
#                 bin_edges,
#                 figure_path,
#                 log_scale=log_scale,
#             )
#         else:
#             raise NotImplementedError(f"unknown backend: {method}")


def correlation_matrix(
    fit_results: fit.FitResults,
    output_path: Union[str, pathlib.Path],
    pruning_threshold: float = 0.0,
) -> None:
    """Draws a correclation matrix.

    Parameters
    ----------
    fit_results : fit.FitResults
        Fit results with correlation matrix and param labels.
    output_path : Union[str, pathlib.Path]
        Directoy where figures are saved.
    pruning_threshold : float
        Minimum correlation of a parameter to have with any other parameter to get included in the plot. Defaults to 0.0.
    """

    # create a matrix that is True if a correlation is below threshold, and True on the
    # diagonal
    below_threshold = np.where(
        np.abs(fit_results.corr_mat) < pruning_threshold, True, False
    )
    np.fill_diagonal(below_threshold, True)

    # entire rows or columns below threshold
    all_below_threshold = np.all(below_threshold, axis=0)

    # rows and columns corresponding to fixed params
    fixed_parameter = np.all(np.equal(fit_results.corr_mat, 0.0), axis=0)

    # delete rows and columns below threshold or where param is fixed
    delete_indices = np.where(np.logical_or(all_below_threshold, fixed_parameter))
    corr_mat = np.delete(
        np.delete(fit_results.corr_mat, delete_indices, axis=1), delete_indices, axis=0
    )
    labels = np.delete(fit_results.labels, delete_indices)

    figure_path = pathlib.Path(output_path) / "correlation_matrix.pdf"
    plotting.correlation_matrix(corr_mat, labels, figure_path)


def pulls(
    fit_results: fit.FitResults,
    output_path: Union[str, pathlib.Path],
    exclude_list: Optional[List[str]] = None,
) -> None:
    """Draws pull plot of parameters and uncertainties.

    Parameters
    ----------
    fit_results : fit.FitResults
        Fit results with correlation matrix and parameter labels.
    output_path : Union[str, pathlib.Path]
        Output path where figures should be saved.
    exclude_list : Optional[List[str]]
        List of parameters to be excluded from plot. Defaults to None.
    """

    figure_path = pathlib.Path(output_path) / "pulls.pdf"
    labels_np = np.asarray(fit_results.labels)

    if exclude_list is None:
        exclude_list = []

    # exclude fixed parameters from pull plot
    exclude_list += [
        label
        for i_np, label in enumerate(labels_np)
        if fit_results.uncertainty[i_np] == 0.0
    ]

    # exclude staterror parameters from pull plot (they are centered at 1)
    exclude_list += [label for label in labels_np if label[0:10] == "staterror_"]

    # exclude unconstrained factors from pull plot
    exclude_list += [
        label
        for i_np, label in enumerate(labels_np)
        if fit_results.types[i_np] == "unconstrained"
    ]

    # filter out parameters
    mask = [True if label not in exclude_list else False for label in labels_np]
    bestfit = fit_results.bestfit[mask]
    uncertainty = fit_results.uncertainty[mask]
    labels_np = labels_np[mask]

    plotting.pulls(bestfit, uncertainty, labels_np, figure_path)
