import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np

from . import configuration
from . import fitter
from . import model_tools
from . import yields
from .helpers import plotting

log = logging.getLogger(__name__)


def _build_figure_name(region_name: str, is_prefit: bool) -> str:
    """Constructs a file name for a figure."""
    figure_name = region_name.replace(" ", "-")
    if is_prefit:
        figure_name += "_" + "prefit"
    else:
        figure_name += "_" + "postfit"
    figure_name += ".pdf"
    return figure_name


def _build_table_name(region_name: str, is_prefit: bool) -> str:
    """Constructs a file name for a table."""
    table_name = region_name.replace(" ", "-")
    if is_prefit:
        table_name += "_" + "prefit"
    else:
        table_name += "_" + "postfit"
    table_name += ".tex"
    return table_name


def _get_binning(region: Dict[str, Any]) -> np.ndarray:
    """Returns the binning to be used in a region."""
    if not region.get("Binning", False):
        raise NotImplementedError("cannot determine binning")

    return np.asarray(region["Binning"])


def yieldsTable(
    spec: Dict[str, Any],
    table_folder: Union[str, pathlib.Path],
    fit_results: Optional[fitter.FitResults] = None,
    signal_name: Optional[str] = None,
) -> None:
    """Creates post-fit yieldstabes.

    Parameters
    ----------
    figure_folder : Union[str, pathlib.Path]
        Directory where to save the figures.
    spec : Dict[str, Any]
        workspace spec in pyhf format.
    fit_results : Optional[fitter.FitResults]
        Fit results including best-fit params and uncertainties as well
        as correlation matrix. Defaults to None, in which case before fit is plotted.
    signal_name : Optional[str]
        Name of a signal process, if present. Will prevent this sample
        to be included in 'total fitted bkg'.
    """

    model, data_combined = model_tools.model_and_data(spec, asimov=False)

    ylds = yields._get_data_yield_uncertainties(spec, fit_results)

    for channel_name in model.config.channels:

        table_path = pathlib.Path(table_folder) / _build_table_name(channel_name, False)

        plotting.yieldsTable(
            channel_name,
            ylds.data[channel_name].size,
            model.config.samples,
            ylds.data[channel_name],
            ylds.yields[channel_name],
            ylds.uncertainties[channel_name],
            table_path,
            signal_name,
        )


def data_MC(
    config: Dict[str, Any],
    spec: Dict[str, Any],
    figure_folder: Union[str, pathlib.Path],
    fit_results: Optional[fitter.FitResults] = None,
    log_scale: Optional[bool] = None,
) -> None:
    """Draws before and after fit data/MC plots from pyhf workspace.

    Parameters
    ----------
    config : Dict[str, Any]
        configuration for the regions
    spec : Dict[str, Any]
        workspace spec in pyhf format.
    figure_folder : Union[str, pathlib.Path]
        Directory where to save the figures.
    fit_results : Optional[fitter.FitResults]
        Fit results including best-fit params and uncertainties as well
        as correlation matrix. Defaults to None, in which case before fit is plotted.
    log_scale : Optional[bool]
        Use log scale for y-axis. Defaults to None, in which case
        it automatically determines what to use.
    """

    model, data_combined = model_tools.model_and_data(spec, with_aux=False)
    ylds = yields._get_data_yield_uncertainties(spec, fit_results)

    if fit_results is not None:
        prefit = False
    else:
        prefit = True

    for channel_name in model.config.channels:  # process channel by channel
        histogram_dict_list = []  # one dict per region/channel

        # get the region dictionary from the config for binning / variable name
        region_dict = configuration.get_region_dict(config, channel_name)
        bin_edges = _get_binning(region_dict)
        variable = region_dict["Variable"]

        for i_sam, sample_name in enumerate(model.config.samples):
            histogram_dict_list.append(
                {
                    "label": sample_name,
                    "isData": False,
                    "yields": ylds.yields[channel_name][i_sam],
                    "variable": variable,
                }
            )

        # add data sample
        histogram_dict_list.append(
            {
                "label": "Data",
                "isData": True,
                "yields": ylds.data[channel_name],
                "variable": variable,
            }
        )

        if prefit:
            figure_path = pathlib.Path(figure_folder) / _build_figure_name(
                channel_name, True
            )
        else:
            figure_path = pathlib.Path(figure_folder) / _build_figure_name(
                channel_name, False
            )
        plotting.data_MC(
            histogram_dict_list,
            np.asarray(ylds.uncertainties[channel_name]),
            bin_edges,
            figure_path,
            log_scale=log_scale,
        )


def correlation_matrix(
    fit_results: fitter.FitResults,
    output_path: Union[str, pathlib.Path],
    pruning_threshold: float = 0.0,
    **kwargs: int,
) -> None:
    """Draws a correlation matrix.

    Parameters
    ----------
    fit_results : fitter.FitResults
        Fit results with correlation matrix and param labels.
    output_path : Union[str, pathlib.Path]
        Directoy where figures are saved.
    pruning_threshold : float
        Minimum correlation of a parameter to have with any other parameter
        to get included in the plot. Defaults to 0.0.
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
    plotting.correlation_matrix(corr_mat, labels, figure_path, **kwargs)


def pulls(
    fit_results: fitter.FitResults,
    output_path: Union[str, pathlib.Path],
    exclude_list: Optional[List[str]] = None,
    include_staterror: bool = True,
) -> None:
    """Draws pull plot of parameters and uncertainties.

    Parameters
    ----------
    fit_results : fitter.FitResults
        Fit results with correlation matrix and parameter labels.
    output_path : Union[str, pathlib.Path]
        Output path where figures should be saved.
    exclude_list : Optional[List[str]]
        List of parameters to be excluded from plot. Defaults to None.
    include_staterror : bool
        Whether or not to include the staterrors.
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

    # exclude unconstrained params from pull plot
    exclude_list += [
        label
        for i_np, label in enumerate(labels_np)
        if fit_results.types[i_np] == "unconstrained"
    ]

    # reinclude NF back into pull plot, because this is what is usually being done
    include_list = [
        label
        for i_np, label in enumerate(labels_np)
        if (fit_results.types[i_np] == "unconstrained")
        or (label[0:10] == "staterror_" and include_staterror)
    ]

    # filter out parameters
    mask = [True if label not in exclude_list else False for label in labels_np]
    bestfit = fit_results.bestfit[mask]
    uncertainty = fit_results.uncertainty[mask]
    labels = labels_np[mask]

    # reinclude params from reinclude list
    mask = [True if label in include_list else False for label in labels_np]
    unconstrained_bestfit = fit_results.bestfit[mask]
    unconstrained_uncertainty = fit_results.uncertainty[mask]
    unconstrained_labels = labels_np[mask]

    # ordering stuff
    labels_lower = np.array([x.lower() if isinstance(x, str) else x for x in labels])
    _order = np.argsort(labels_lower)
    bestfit = bestfit[_order]
    uncertainty = uncertainty[_order]
    labels = labels[_order]

    unconstrained_labels_lower = np.array(
        [x.lower() if isinstance(x, str) else x for x in unconstrained_labels]
    )
    _order = np.argsort(unconstrained_labels_lower)
    unconstrained_bestfit = unconstrained_bestfit[_order]
    unconstrained_uncertainty = unconstrained_uncertainty[_order]
    unconstrained_labels = unconstrained_labels[_order]

    plotting.pulls(
        bestfit,
        uncertainty,
        labels,
        unconstrained_bestfit,
        unconstrained_uncertainty,
        unconstrained_labels,
        figure_path,
    )
