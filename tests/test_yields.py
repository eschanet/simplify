import awkward1 as ak
import numpy as np
import pytest

from simplify import fitter
from simplify import model_tools
from simplify import yields as yld


def test_FitResults():

    regions = ["region_A"]
    samples = ["sample_A"]
    yields = {"region_A": np.asarray([1.0])}
    uncertainties = {"region_A": ak.from_iter([0.1])}
    data = {"region_A": np.asarray([1.0])}

    y = yld.Yields(regions, samples, yields, uncertainties, data)
    for region, _ in yields.items():
        assert np.allclose(y.yields[region], yields[region])
        assert np.allclose(
            ak.to_list(y.uncertainties[region]), ak.to_list(uncertainties[region])
        )
        assert np.allclose(y.data[region], data[region])
    assert y.regions == regions
    assert y.samples == samples


def test___pdgRound():

    assert yld._pdgRound(0, 0) == ("0.0", "0.0")
    assert yld._pdgRound(0, 99.99) == ("0", "100")
    assert yld._pdgRound(1.23, 1.0) == ("1.2", "1.0")
    assert yld._pdgRound(8.232, 5.123) == ("8", "5")
    assert yld._pdgRound(955.2, 5.123) == ("955", "5")
    assert yld._pdgRound(100.55, 0.99698) == ("100.5", "1.0")

    assert yld._pdgRound(123.456, 0.0) == ("120", "0")
    assert yld._pdgRound(123.456, 12.34) == ("123", "12")
    assert yld._pdgRound(123.456, -12.34) == ("123", "-12")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test__get_data_yield_uncertainties(example_spec):
    model, data = model_tools.model_and_data(example_spec)
    fit_results = fitter.fit(model, data)

    y = yld._get_data_yield_uncertainties(example_spec, fit_results)
    for region in y.regions:
        assert np.allclose(y.yields[region], np.asarray([690.99844915]))
        assert np.allclose(ak.to_list(y.uncertainties[region]), [26.278787667809468])
        assert np.allclose(y.data[region], np.asarray([691]))
    assert y.regions == ['SR']
    assert y.samples == ['signal']

    y = yld._get_data_yield_uncertainties(example_spec, None)
    for region in y.regions:
        assert np.allclose(y.yields[region], np.asarray([112.429786]))
        assert np.allclose(ak.to_list(y.uncertainties[region]), [0.0])
        assert np.allclose(y.data[region], np.asarray([691]))
    assert y.regions == ['SR']
    assert y.samples == ['signal']


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_yields(example_spec):
    model, data = model_tools.model_and_data(example_spec)
    fit_results = fitter.fit(model, data)

    y = yld.get_yields(example_spec, fit_results)
    for region in y.regions:
        assert np.allclose(y.yields[region], np.asarray([690.99844915]))
        assert np.allclose(ak.to_list(y.uncertainties[region]), [26.278787667809468])
        assert np.allclose(y.data[region], np.asarray([691]))
    assert y.regions == ['SR']
    assert y.samples == ['signal']
