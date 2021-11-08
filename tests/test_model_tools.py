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
    model, data = model_tools.model_and_data(pyhf.Workspace(example_spec), asimov=True)
    assert model.spec["channels"] == example_spec["channels"]
    assert model.config.modifier_settings == {
        "normsys": {"interpcode": "code4"},
        "histosys": {"interpcode": "code4p"},
    }
    assert data == [112.429786, 1.0]

    # test handing a workspace instead of JSON
    model, data = model_tools.model_and_data(pyhf.Workspace(example_spec))
    assert model.spec["channels"] == example_spec["channels"]
    assert model.config.modifier_settings == {
        "normsys": {"interpcode": "code4"},
        "histosys": {"interpcode": "code4p"},
    }
    assert data == [691, 1.0]

    # without auxdata
    model, data = model_tools.model_and_data(example_spec, include_auxdata=False)
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


def test__get_channel_bounds_indices(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    indices = model_tools._get_channel_bounds_indices(model)
    assert indices == []

    model = pyhf.Workspace(example_spec_multibin).model()
    indices = model_tools._get_channel_bounds_indices(model)
    assert indices == [2]

    # add extra channel to model to test three channels (two indices needed)
    three_channel_model = copy.deepcopy(example_spec_multibin)
    extra_channel = copy.deepcopy(three_channel_model["channels"][0])
    extra_channel["name"] = "region_3"
    extra_channel["samples"][0]["modifiers"][0]["name"] = "staterror_region_3"
    three_channel_model["channels"].append(extra_channel)
    three_channel_model["observations"].append({"data": [35, 8], "name": "region_3"})
    model = pyhf.Workspace(three_channel_model).model()
    indices = model_tools._get_channel_bounds_indices(model)
    assert indices == [2, 3]


def test_calculate_std(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    parameters = np.asarray([1.05, 0.95])
    uncertainty = np.asarray([0.1, 0.1])
    corr_mat = np.asarray([[1.0, 0.2], [0.2, 1.0]])

    total_std = model_tools.calculate_std(model, parameters, uncertainty, corr_mat)
    expected_std = [[17.4320561320614]]
    assert np.allclose(ak.to_list(total_std), expected_std)

    # pre-fit
    parameters = np.asarray([1.0, 1.0])
    uncertainty = np.asarray([0.0495665682, 0.0])
    diag_corr_mat = np.diag([1.0, 1.0])
    total_std = model_tools.calculate_std(model, parameters, uncertainty, diag_corr_mat)
    expected_std = [[5.572758655480406]]  # the staterror
    assert np.allclose(ak.to_list(total_std), expected_std)

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
    total_std = model_tools.calculate_std(model, parameters, uncertainty, corr_mat)
    expected_std = [[12.889685799118613, 2.6730057987217317], [3.469221814759039]]
    for i_reg in range(2):
        assert np.allclose(ak.to_list(total_std[i_reg]), expected_std[i_reg])
