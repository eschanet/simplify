import pyhf
import numpy as np

from typing import Any, Dict, List, Tuple

import logging
from . import logger
log = logging.getLogger(__name__)


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
    workspace = pyhf.Workspace(spec)
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




def build_Asimov_data(model: pyhf.Model, with_aux: bool = True) -> List[float]:
    pass
