import copy
import pathlib

from matplotlib.testing.compare import compare_images
import numpy as np

from simplify.helpers import plotting


def test_yieldsTable(tmp_path):

    fname = pathlib.Path(tmp_path / "table.tex")

    region_name = "SR_A"
    nbins = 2
    samples = ["bkg_1", "bkg_2", "signal"]
    data = np.asarray([15, 10])
    yields = np.asarray([[4.5, 1.5], [2.5, 1.5], [8.0, 6.0]])
    uncertainties = np.asarray([1.8, 3.4])

    # standalone with signal specified
    plotting.yieldsTable(
        region_name,
        nbins,
        samples,
        data,
        yields,
        uncertainties,
        fname,
        signal_name="signal",
        standalone=True,
    )

    assert [
        row.strip()
        for row in open("tests/helpers/reference/yieldstable_standalone.tex")
    ] == [row.strip() for row in open(fname)]

    region_name = "SR_A"
    nbins = 1
    samples = ["bkg_1", "bkg_2", "signal"]
    data = np.asarray([15])
    yields = np.asarray([[4.5], [2.5], [8.0]])
    uncertainties = np.asarray([1.8])

    # not standalone and only one bin
    plotting.yieldsTable(
        region_name,
        nbins,
        samples,
        data,
        yields,
        uncertainties,
        fname,
        signal_name=None,
        standalone=False,
    )

    assert [row.strip() for row in open("tests/helpers/reference/yieldstable.tex")] == [
        row.strip() for row in open(fname)
    ]


def test_correlation_matrix(tmp_path):
    fname = pathlib.Path(tmp_path / "fig.pdf")
    # one parameter is below threshold so no text is shown for it on the plot
    corr_mat = np.asarray(
        [[1.0, 0.375, 0.002], [0.375, 1.0, -0.55], [0.002, -0.55, 1.0]]
    )
    labels = ["param_a", "param_b", "param_c"]
    plotting.correlation_matrix(corr_mat, labels, fname)
    assert (
        compare_images("tests/helpers/reference/correlation_matrix.pdf", fname, 0)
        is None
    )


def test_pulls(tmp_path):
    fname = pathlib.Path(tmp_path / "fig.pdf")
    bestfit_constrained = np.asarray([-0.2, 0.0, 0.1])
    uncertainty_constrained = np.asarray([0.9, 1.0, 0.7])
    labels_constrained = np.asarray(["param_d", "param_e", "param_f"])
    bestfit_unconstrained = np.asarray([1.05, 0.8, 1.3])
    uncertainty_unconstrained = np.asarray([0.2, 0.5, 0.1])
    labels_unconstrained = np.asarray(["param_a", "param_b", "param_c"])
    plotting.pulls(
        bestfit_constrained,
        uncertainty_constrained,
        labels_constrained,
        bestfit_unconstrained,
        uncertainty_unconstrained,
        labels_unconstrained,
        fname,
    )
    assert compare_images("tests/helpers/reference/pulls.pdf", fname, 0) is None


def test_data_MC(tmp_path):
    fname = pathlib.Path(tmp_path / "fig.pdf")
    histo_dict_list = [
        {
            "label": "Background",
            "isData": False,
            "yields": np.asarray([12.5, 14]),
            "variable": "x",
        },
        {
            "label": "Signal",
            "isData": False,
            "yields": np.asarray([2, 5]),
            "variable": "x",
        },
        {
            "label": "Data",
            "isData": True,
            "yields": np.asarray([13, 15]),
            "variable": "x",
        },
    ]
    total_model_unc = np.sqrt([0.17, 0.29])
    bin_edges = np.asarray([1, 2, 3])
    plotting.data_MC(histo_dict_list, total_model_unc, bin_edges, fname)
    assert compare_images("tests/helpers/reference/data_MC.pdf", fname, 0) is None
    fname.unlink()  # delete figure

    histo_dict_list_log = copy.deepcopy(histo_dict_list)
    histo_dict_list_log[0]["yields"] = np.asarray([2000, 14])
    histo_dict_list_log[2]["yields"] = np.asarray([2010, 15])
    total_model_unc_log = np.asarray([50, 1.5])
    fname_log = pathlib.Path(tmp_path / "fig_log.pdf")

    # automatic log scale
    plotting.data_MC(histo_dict_list_log, total_model_unc_log, bin_edges, fname)
    assert (
        compare_images("tests/helpers/reference/data_MC_log.pdf", fname_log, 0) is None
    )
    fname_log.unlink()

    # linear scale forced
    plotting.data_MC(
        histo_dict_list, total_model_unc, bin_edges, fname, log_scale=False
    )
    assert compare_images("tests/helpers/reference/data_MC.pdf", fname, 0) is None
    fname.unlink()

    # log scale forced
    plotting.data_MC(
        histo_dict_list_log, total_model_unc_log, bin_edges, fname, log_scale=True
    )
    assert (
        compare_images("tests/helpers/reference/data_MC_log.pdf", fname_log, 0) is None
    )
    fname_log.unlink()
