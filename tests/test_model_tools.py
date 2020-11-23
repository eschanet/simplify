import copy

import awkward1 as ak
import numpy as np
import pyhf

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

    # TODO: Need to test overloaded method as well

    # test handing a workspace instead of JSON
    model, data = model_tools.model_and_data(pyhf.Workspace(example_spec))
    assert model.spec["channels"] == example_spec["channels"]
    assert model.config.modifier_settings == {
        "normsys": {"interpcode": "code4"},
        "histosys": {"interpcode": "code4p"},
    }
    assert data == [691, 1.0]

    # without auxdata
    model, data = model_tools.model_and_data(example_spec, with_aux=False)
    assert data == [691]


def test_get_parameter_names(example_spec):
    model = pyhf.Workspace(example_spec).model()
    labels = model_tools.get_parameter_names(model)
    assert labels == ["staterror_SR", "mu_Sig"]


def test_get_parameter_types(example_spec):
    model = pyhf.Workspace(example_spec).model()
    labels = model_tools.get_parameter_types(model)
    assert labels == ["constrained", "unconstrained"]


def test_get_prefit_uncertainties(
    example_spec, example_spec_multibin, example_spec_shapefactor
):
    model = pyhf.Workspace(example_spec).model()
    uncertainties = model_tools.get_prefit_uncertainties(model)
    assert np.allclose(uncertainties, [0.0, 0.0])

    model = pyhf.Workspace(example_spec_multibin).model()
    uncertainties = model_tools.get_prefit_uncertainties(model)
    assert np.allclose(uncertainties, [0.175, 0.375, 0.0, 0.2])

    model = pyhf.Workspace(example_spec_shapefactor).model()
    uncertainties = model_tools.get_prefit_uncertainties(model)
    assert np.allclose(uncertainties, [0.0, 0.0, 0.0])


def test__get_channel_boundary_indices(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    indices = model_tools._get_channel_boundary_indices(model)
    assert indices == []

    model = pyhf.Workspace(example_spec_multibin).model()
    indices = model_tools._get_channel_boundary_indices(model)
    assert indices == [2]

    # add extra channel to model to test three channels (two indices needed)
    three_channel_model = copy.deepcopy(example_spec_multibin)
    extra_channel = copy.deepcopy(three_channel_model["channels"][0])
    extra_channel["name"] = "region_3"
    extra_channel["samples"][0]["modifiers"][0]["name"] = "staterror_region_3"
    three_channel_model["channels"].append(extra_channel)
    three_channel_model["observations"].append({"data": [35, 8], "name": "region_3"})
    model = pyhf.Workspace(three_channel_model).model()
    indices = model_tools._get_channel_boundary_indices(model)
    assert indices == [2, 3]


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
    expected_stdev = [[12.889685799118613, 2.6730057987217317], [3.469221814759039]]
    for i_reg in range(2):
        assert np.allclose(ak.to_list(total_stdev[i_reg]), expected_stdev[i_reg])
