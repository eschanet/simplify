import numpy as np
import pyhf
import awkward1 as ak

import pytest

from simplify import model_tools


def test_model_and_data(example_spec):
    model, data = model_tools.model_and_data(example_spec)
    assert model.spec["channels"] == example_spec["channels"]
    assert model.config.modifier_settings == {
        "normsys": {"interpcode": "code4"},
        "histosys": {"interpcode": "code4p"},
    }
    assert data == [691, 1.0]

    # requesting Asimov dataset
    # TODO: request asimov dataset by setting asimove=True
    # TODO: should return [112.429786, 1.0]

    # without auxdata
    model, data = model_tools.model_and_data(example_spec, with_aux=False)
    assert data == [691]


def test_get_parameter_names(example_spec):
    model = pyhf.Workspace(example_spec).model()
    labels = model_tools.get_parameter_names(model)
    assert labels == ["staterror_SR", "mu_Sig"]


def test_calculate_stdev(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    parameters = np.asarray([1.05, 0.95])
    uncertainty = np.asarray([0.1, 0.1])
    corr_mat = np.asarray([[1.0, 0.2], [0.2, 1.0]])

    total_stdev = model_tools.calculate_stdev(model, parameters, uncertainty, corr_mat)
    expected_stdev = [[17.4320561320614]]
    assert np.allclose(ak.to_list(total_stdev), expected_stdev)

    # pre-fit
    parameters = np.asarray([1.0, 1.0])
    uncertainty = np.asarray([0.0495665682, 0.0])
    diag_corr_mat = np.diag([1.0, 1.0])
    total_stdev = model_tools.calculate_stdev(
        model, parameters, uncertainty, diag_corr_mat
    )
    expected_stdev = [[5.572758655480406]]  # the staterror
    assert np.allclose(ak.to_list(total_stdev), expected_stdev)

    # multiple channels, bins, staterrors
    model = pyhf.Workspace(example_spec_multibin).model()
    parameters = np.asarray([0.9, 1.05, 1.3, 0.95])
    uncertainty = np.asarray([0.1, 0.05, 0.3, 0.1])
    corr_mat = np.asarray(
        [
            [1.0, 0.1, 0.2, 0.1],
            [0.1, 1.0, 0.2, 0.3],
            [0.2, 0.2, 1.0, 0.3],
            [0.1, 0.3, 0.3, 1.0],
        ]
    )
    total_stdev = model_tools.calculate_stdev(model, parameters, uncertainty, corr_mat)
    expected_stdev = [[12.889685799118613, 2.6730057987217317], [3.8161439962349433]]
    for i_reg in range(2):
        assert np.allclose(ak.to_list(total_stdev[i_reg]), expected_stdev[i_reg])
