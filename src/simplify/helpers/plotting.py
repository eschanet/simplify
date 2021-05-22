import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


log = logging.getLogger(__name__)


def yieldsTable(
    region_name: str,
    nbins: int,
    samples: List[str],
    data: np.ndarray,
    yields: np.ndarray,
    uncertainties: np.ndarray,
    figure_path: pathlib.Path,
    signal_name: Optional[str] = None,
    standalone: Optional[bool] = True,
) -> None:
    """Print a yieldstable in Latex format"""

    if standalone:
        header = r'''\documentclass{standalone}
\usepackage{longtable}
\usepackage{booktabs}
\newcommand\MyHead[2]{%
\multicolumn{1}{l}{\parbox{#1}{\centering #2}}
}
\begin{document}
        '''
        footer = r'''
\end{document}
'''
    else:
        header = ''
        footer = ''

    columns = "l"
    columns += "".join(["c"] * nbins)
    if nbins > 1:
        columns += "c"

    region_name = region_name.replace('_', r'\_')
    column_names = region_name
    if nbins > 0:
        for i_bin in range(nbins):
            column_names += r' & %s\_bin%i' % (region_name, i_bin)

    if signal_name:
        # signal_index = samples.index(signal_name)
        signal_row = np.array(
            [i for i in range(yields.shape[0]) if not i == samples.index(signal_name)]
        )
        bkgOnly_yields = yields[
            signal_row[
                :,
            ],
            :,
        ]  # np.delete(yields, signal_index, 0)

    else:
        bkgOnly_yields = yields

    # FIXME: this still has signal uncertainties
    # in total uncertainty, which is not right!!!!
    # get total region first, then do the bins
    data_line = "Observed events & ${}$".format(np.sum(data))
    total_sm = r'Fitted bkg events & ${:8.3f} \pm {:8.3f}'.format(
        np.sum(bkgOnly_yields), np.sqrt(np.sum(uncertainties ** 2))
    )

    # FIXME: same as above, dooh...
    if nbins > 1:
        for i_bin in range(nbins):
            data_line += " & ${}$".format(data[i_bin])
            total_sm += r' & ${:8.3f} \pm {:8.3f}$'.format(
                np.sum(bkgOnly_yields[:, i_bin]), uncertainties[i_bin]
            )

    data_line += r'''\\
'''
    total_sm += r'''\\
'''

    main = ''
    for i_sample, sample in enumerate(samples):
        main += "Fitted {} events & ${:8.3f}$".format(
            sample.replace("_", r'\_'), np.sum(yields[i_sample, :])
        )
        if nbins > 1:
            for i_bin in range(nbins):
                main += " & ${:8.3f}$".format(yields[i_sample, i_bin])

        main += r'''\\
'''

    content = r'''%
{}
\begin{{tabular}}{{{}}}
\toprule
{} \\
\midrule
{}
\midrule
{}
\midrule
{}
\bottomrule
\end{{tabular}}
{}
'''.format(
        header,
        columns,
        column_names,
        data_line,
        total_sm,
        main,
        footer,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with open(figure_path, 'w') as f:
        f.write(content)


def data_MC(
    histogram_dict_list: List[Dict[str, Any]],
    total_model_unc: np.ndarray,
    bin_edges: np.ndarray,
    figure_path: pathlib.Path,
    log_scale: Optional[bool] = None,
) -> None:
    """Draw a data/MC histogram with uncertainty bands and ratio panel."""
    mc_histograms_yields = []
    mc_labels = []
    for h in histogram_dict_list:
        if h["isData"]:
            data_histogram_yields = h["yields"]
            data_histogram_stdev = np.sqrt(data_histogram_yields)
            data_label = h["label"]
        else:
            mc_histograms_yields.append(h["yields"])
            mc_labels.append(h["label"])

    mpl.style.use("seaborn-colorblind")

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # increase font sizes
    for item in (
        [ax1.yaxis.label, ax2.xaxis.label, ax2.yaxis.label]
        + ax1.get_yticklabels()
        + ax2.get_xticklabels()
        + ax2.get_yticklabels()
    ):
        item.set_fontsize("large")

    # minor ticks on all axes
    for axis in [ax1.xaxis, ax1.yaxis, ax2.xaxis, ax2.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    # plot MC stacked together
    total_yield = np.zeros_like(mc_histograms_yields[0])
    bin_right_edges = bin_edges[1:]
    bin_left_edges = bin_edges[:-1]
    bin_width = bin_right_edges - bin_left_edges
    bin_centers = 0.5 * (bin_left_edges + bin_right_edges)
    mc_containers = []
    for mc_sample_yield in mc_histograms_yields:
        mc_container = ax1.bar(
            bin_centers,
            mc_sample_yield,
            width=bin_width,
            bottom=total_yield,
        )
        mc_containers.append(mc_container)

        # add a black line on top of each sample
        line_x = [y for y in bin_edges for _ in range(2)][1:-1]
        line_y = [y for y in (mc_sample_yield + total_yield) for _ in range(2)]
        ax1.plot(line_x, line_y, "-", color="black", linewidth=0.5)

        total_yield += mc_sample_yield

    # add total MC uncertainty
    mc_unc_container = ax1.bar(
        bin_centers,
        2 * total_model_unc,
        width=bin_width,
        bottom=total_yield - total_model_unc,
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    # plot data
    data_container = ax1.errorbar(
        bin_centers,
        data_histogram_yields,
        yerr=data_histogram_stdev,
        fmt="o",
        color="k",
    )

    # ratio plot
    ax2.plot(
        [bin_left_edges[0], bin_right_edges[-1]],
        [1, 1],
        "--",
        color="black",
        linewidth=1,
    )  # reference line along y=1

    # add uncertainty band around y=1
    rel_mc_unc = total_model_unc / total_yield
    ax2.bar(
        bin_centers,
        2 * rel_mc_unc,
        width=bin_width,
        bottom=1 - rel_mc_unc,
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    # data in ratio plot
    data_model_ratio = data_histogram_yields / total_yield
    data_model_ratio_unc = data_histogram_stdev / total_yield
    ax2.errorbar(
        bin_centers, data_model_ratio, yerr=data_model_ratio_unc, fmt="o", color="k"
    )

    # get the highest single bin yield, from the sum of MC or data
    y_max = max(
        np.max(total_yield),
        np.max([h["yields"] for h in histogram_dict_list if h["isData"]]),
    )
    # lowest MC yield in single bin (not considering empty bins)
    y_min = np.min(total_yield[np.nonzero(total_yield)])

    # determine scale setting, unless it is provided
    if log_scale is None:
        # if yields vary over more than 2 orders of magnitude, set y-axis to log scale
        log_scale = (y_max / y_min) > 100

    # set vertical axis scale and limits
    if log_scale:
        # use log scale
        ax1.set_yscale("log")
        ax1.set_ylim([y_min / 10, y_max * 10])
        # add "_log" to the figure name
        figure_path = figure_path.with_name(
            figure_path.stem + "_log" + figure_path.suffix
        )
    else:
        # do not use log scale
        ax1.set_ylim([0, y_max * 1.5])  # 50% headroom

    # MC contributions in inverse order, such that first legend entry corresponds to
    # the last (highest) contribution to the stack
    all_containers = mc_containers[::-1] + [mc_unc_container, data_container]
    all_labels = mc_labels[::-1] + ["Uncertainty", data_label]
    ax1.legend(all_containers, all_labels, frameon=False, fontsize="large")

    ax1.set_xlim(bin_left_edges[0], bin_right_edges[-1])
    ax1.set_ylabel("events")
    ax1.set_xticklabels([])
    ax1.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
    ax1.tick_params(direction="in", top=True, right=True, which="both")

    ax2.set_xlim(bin_left_edges[0], bin_right_edges[-1])
    ax2.set_ylim([0.5, 1.5])
    ax2.set_xlabel(histogram_dict_list[0]["variable"])
    ax2.set_ylabel("data / model")
    ax2.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax2.set_yticklabels([0.5, 0.75, 1.0, 1.25, ""])
    ax2.tick_params(axis="both", which="major", pad=8)
    ax2.tick_params(direction="in", top=True, right=True, which="both")

    fig.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)


def correlation_matrix(
    corr_mat: np.ndarray,
    labels: Union[List[str], np.ndarray],
    figure_path: pathlib.Path,
    **kwargs: int,
) -> None:
    """Draw a correlation matrix"""

    # rounding for test in CI to match reference
    fig, ax = plt.subplots(
        figsize=(round(5 + len(labels) / 1.6, 1), round(3 + len(labels) / 1.6, 1)),
        dpi=100,
    )
    vmin = kwargs.get('vmin', -1)
    vmax = kwargs.get('vmax', 1)
    cmap = kwargs.get('cmap', 'RdBu')
    tmin = kwargs.get('tmin', 0.005)

    im = ax.imshow(corr_mat, vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")

    fig.colorbar(im, ax=ax)
    ax.set_aspect("auto")  # to get colorbar aligned with matrix
    fig.tight_layout()

    # add correlation as text
    for (j, i), corr in np.ndenumerate(corr_mat):
        text_color = "white" if abs(corr_mat[j, i]) > 0.75 else "black"
        if abs(corr) > tmin:
            ax.text(i, j, f"{corr:.2f}", ha="center", va="center", color=text_color)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)


def pulls(
    bestfit_constrained: np.ndarray,
    uncertainty_constrained: np.ndarray,
    labels_constrained: Union[List[str], np.ndarray],
    bestfit_unconstrained: np.ndarray,
    uncertainty_unconstrained: np.ndarray,
    labels_unconstrained: Union[List[str], np.ndarray],
    figure_path: pathlib.Path,
) -> None:
    """Draws a pull plot."""
    num_pars_constrained = len(bestfit_constrained)
    num_pars_unconstrained = len(bestfit_unconstrained)
    num_pars = num_pars_constrained + num_pars_unconstrained

    y_positions = np.arange(num_pars)[::-1]
    y_pos_constrained = np.arange(num_pars_constrained)[::-1]
    y_pos_unconstrained = np.arange(y_pos_constrained[0] + 1, num_pars)[::-1]
    fig, ax = plt.subplots(figsize=(6, 1 + num_pars / 4), dpi=100)

    ax2 = ax.twiny()

    ax.errorbar(
        bestfit_constrained,
        y_pos_constrained,
        xerr=uncertainty_constrained,
        fmt="o",
        color="black",
    )
    ax2.errorbar(
        bestfit_unconstrained,
        y_pos_unconstrained,
        xerr=uncertainty_unconstrained,
        fmt="o",
        color="blue",
    )

    ax.fill_between([-2, 2], -0.5, num_pars - 0.5, color="#fff9ab")
    ax.fill_between([-1, 1], -0.5, num_pars - 0.5, color="#a4eb9b")
    ax.vlines(0, -0.5, num_pars - 0.5, linestyles="dotted", color="black")
    # ax.hlines(0, -0.5, len(num_pars) - 0.5, linestyles="dotted", color="black")

    ax.set_xlim([-3, 3])
    ax.set_xlabel(r"$\left(\hat{\theta} - \theta_0\right) / \Delta \theta$")
    ax.set_ylim([-0.5, num_pars - 0.5])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(np.append(labels_unconstrained, labels_constrained, axis=0))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())  # minor ticks
    ax.tick_params(axis="both", which="major", pad=8)
    ax.tick_params(direction="in", top=True, right=True, which="both")

    ax2.set_xlim([-0.5, 2.5])
    ax2.set_xlabel(r"$\gamma / \mu$")
    ax2.spines['top'].set_color('blue')
    ax2.xaxis.label.set_color('blue')
    ax2.tick_params(axis='x', colors='blue')

    fig.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)
