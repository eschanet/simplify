import pathlib
from unittest import mock

import awkward1 as ak
import numpy as np
import pyhf
import pytest

from simplify import fitter
from simplify import plot


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("SR", True), "SR_prefit.pdf"),
        (("SR", False), "SR_postfit.pdf"),
        (("SR 1", True), "SR-1_prefit.pdf"),
        (("SR 1", False), "SR-1_postfit.pdf"),
    ],
)
def test__build_figure_name(test_input, expected):
    assert plot._build_figure_name(*test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("SR", True), "SR_prefit.tex"),
        (("SR", False), "SR_postfit.tex"),
        (("SR 1", True), "SR-1_prefit.tex"),
        (("SR 1", False), "SR-1_postfit.tex"),
    ],
)
def test__build_table_name(test_input, expected):
    assert plot._build_table_name(*test_input) == expected


def test__get_binning():
    np.testing.assert_equal(plot._get_binning({"Binning": [1, 2, 3]}), [1, 2, 3])
    with pytest.raises(NotImplementedError, match="cannot determine binning"):
        plot._get_binning({})


@mock.patch("simplify.helpers.plotting.data_MC")
@mock.patch("simplify.plot._get_binning", return_value=np.asarray([1, 2]))
@mock.patch(
    "simplify.configuration.get_region_dict",
    return_value={"Name": "region", "Variable": "x"},
)
@mock.patch("simplify.model_tools.calculate_stdev", return_value=ak.from_iter([[0.3]]))
@mock.patch(
    "simplify.model_tools.get_prefit_uncertainties",
    return_value=(ak.from_iter([0.04956657, 0.0])),
)
@mock.patch(
    "simplify.model_tools.get_asimov_parameters",
    return_value=([1.0, 1.0]),
)
def test_data_MC(
    mock_asimov, mock_unc, mock_stdev, mock_dict, mock_bins, mock_draw, example_spec
):
    config = {}
    figure_folder = "tmp"
    model_spec = pyhf.Workspace(example_spec).model().spec

    plot.data_MC(config, example_spec, figure_folder=figure_folder)

    # Asimov parameters and prefit uncertainties
    assert mock_asimov.call_count == 1
    assert mock_asimov.call_args_list[0][0][0].spec == model_spec
    assert mock_unc.call_count == 1
    assert mock_unc.call_args_list[0][0][0].spec == model_spec

    # call to stdev calculation
    assert mock_stdev.call_count == 1
    assert mock_stdev.call_args_list[0][0][0].spec == model_spec
    assert np.allclose(mock_stdev.call_args_list[0][0][1], [1.0, 1.0])
    assert np.allclose(
        ak.to_numpy(mock_stdev.call_args_list[0][0][2]), [0.04956657, 0.0]
    )
    assert np.allclose(
        mock_stdev.call_args_list[0][0][3], np.asarray([[1.0, 0.0], [0.0, 1.0]])
    )
    assert mock_stdev.call_args_list[0][1] == {}

    assert mock_dict.call_args_list == [[(config, "SR"), {}]]
    assert mock_bins.call_args_list == [[({"Name": "region", "Variable": "x"},), {}]]

    expected_histograms = [
        {
            "label": "signal",
            "isData": False,
            "yields": np.asarray([112.429786]),
            "variable": "x",
        },
        {
            "label": "Data",
            "isData": True,
            "yields": np.asarray([691]),
            "variable": "x",
        },
    ]
    assert mock_draw.call_count == 1
    assert mock_draw.call_args_list[0][0][0] == expected_histograms
    assert np.allclose(mock_draw.call_args_list[0][0][1], np.asarray([0.3]))
    assert np.allclose(mock_draw.call_args_list[0][0][2], np.asarray([1, 2]))
    assert mock_draw.call_args_list[0][0][3] == pathlib.Path("tmp/SR_prefit.pdf")
    assert mock_draw.call_args_list[0][1] == {"log_scale": None}

    # post-fit plot and custom scale
    fit_results = fitter.FitResults(
        np.asarray([1.01, 1.1]),
        np.asarray([0.03, 0.1]),
        [],
        [],
        np.asarray([[1.0, 0.2], [0.2, 1.0]]),
        0.0,
    )
    plot.data_MC(
        config,
        example_spec,
        figure_folder=figure_folder,
        fit_results=fit_results,
        log_scale=False,
    )

    assert mock_asimov.call_count == 1  # no new call

    # call to stdev calculation
    assert mock_stdev.call_count == 2
    assert mock_stdev.call_args_list[1][0][0].spec == model_spec
    assert np.allclose(mock_stdev.call_args_list[1][0][1], [1.01, 1.1])
    assert np.allclose(mock_stdev.call_args_list[1][0][2], [0.03, 0.1])
    assert np.allclose(
        mock_stdev.call_args_list[1][0][3], np.asarray([[1.0, 0.2], [0.2, 1.0]])
    )
    assert mock_stdev.call_args_list[1][1] == {}

    assert mock_draw.call_count == 2
    # yield at best-fit point is different from pre-fit
    assert np.allclose(mock_draw.call_args_list[1][0][0][0]["yields"], 124.90949225)
    assert np.allclose(mock_draw.call_args_list[1][0][1], np.asarray([0.3]))
    assert np.allclose(mock_draw.call_args_list[1][0][2], np.asarray([1, 2]))
    assert mock_draw.call_args_list[1][0][3] == pathlib.Path("tmp/SR_postfit.pdf")
    assert mock_draw.call_args_list[1][1] == {"log_scale": False}
