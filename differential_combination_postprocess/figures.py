import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

hep.style.use("CMS")
from itertools import cycle
from copy import deepcopy

from .cosmetics import rainbow, observable_specs, category_specs, fit_type_colors
from .shapes import ObservableShapeSM

# Silence matplotlib warnings for Christ sake
import warnings

warnings.filterwarnings("ignore", module="matplotlib")

import logging

logger = logging.getLogger(__name__)


class Figure:
    def __init__(self):
        self.fig, _ = plt.subplots()
        self.output_name = "prototype"

    def dump(self, output_dir):
        # Dump the image in multiple formats
        self.fig.savefig(
            "{}/{}.png".format(output_dir, self.output_name), bbox_inches="tight"
        )
        self.fig.savefig(
            "{}/{}.pdf".format(output_dir, self.output_name), bbox_inches="tight"
        )


class XSNLLsPerPOI:
    """
    Remember that this breaks the convention, since one figure per POI is created
    """

    def __init__(self, subcategory_spectra, debug=False):
        self.figures = []
        nominal_spectrum = list(subcategory_spectra.values())[0]
        pois = nominal_spectrum.scans.keys()

        for poi in pois:
            scans = []
            for subcategory, spectrum in subcategory_spectra.items():
                scans.append((subcategory, spectrum.scans[poi]))

            fig, ax = plt.subplots()
            output_name = (
                f"NLLs_{nominal_spectrum.variable}_{nominal_spectrum.category}_{poi}"
            )
            if debug:
                output_name += "_debug"

            # Set labels
            ax.set_xlabel(poi)
            ax.set_ylabel("-2$\Delta$lnL")

            # Set limits
            ax.set_ylim(0.0, 8.0)

            # Draw horizontal line at 1
            ax.axhline(1.0, color="k", linestyle="--")

            # Draw horizontal line at 4
            ax.axhline(4.0, color="k", linestyle="--")

            # Draw all the NLLs with different colors
            for scan_tpl, color in zip(scans, fit_type_colors):
                ax = scan_tpl[1].plot(ax, color, label=scan_tpl[0])
                if debug:
                    ax = scan_tpl[1].plot_original_points(
                        ax, color, label=f"{scan_tpl[0]} (original)"
                    )

            # Add note with minimum and uncertainties
            nomstring = f"{nominal_spectrum.scans[poi].minimum[0]:.3f}"
            upstring = f"{nominal_spectrum.scans[poi].up68_unc:.3f}"
            upstring = "{+" + upstring + "}"
            downstring = f"{nominal_spectrum.scans[poi].down68_unc:.3f}"
            downstring = "{-" + downstring + "}"
            ax.text(
                nominal_spectrum.scans[poi].minimum[0],
                2.5,
                f"{poi} = ${nomstring}^{upstring}_{downstring}$",
                fontsize=14,
                ha="center",
                va="center",
            )

            # Legend
            ax.legend(loc="upper center", prop={"size": 10}, ncol=4)
            hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=138, ax=ax)

            self.figures.append((fig, ax, output_name))

    def dump(self, output_dir):
        for fig, ax, output_name in self.figures:
            # Dump the image in multiple formats
            fig.savefig(
                "{}/{}.png".format(output_dir, output_name), bbox_inches="tight"
            )
            fig.savefig(
                "{}/{}.pdf".format(output_dir, output_name), bbox_inches="tight"
            )


class XSNLLsPerPOI_Full(XSNLLsPerPOI):
    def __init__(self, subcategory_spectra):
        self.figures = []
        nominal_spectrum = list(subcategory_spectra.values())[0]
        pois = nominal_spectrum.scans.keys()

        for poi in pois:
            scans = []
            for subcategory, spectrum in subcategory_spectra.items():
                scans.append((subcategory, spectrum.scans[poi]))

            fig, ax = plt.subplots()
            output_name = f"Full_NLLs_{nominal_spectrum.variable}_{nominal_spectrum.category}_{poi}"

            # Set labels
            ax.set_xlabel(poi)
            ax.set_ylabel("-$\Delta$lnL")

            # Draw all the NLLs with different colors
            rainbow_iter = cycle(rainbow)
            for scan_tpl in scans:
                color = next(rainbow_iter)
                ax = scan_tpl[1].plot_original_points(
                    ax, color, label=scan_tpl[0], for_single_plot=True
                )

            # Legend
            ax.legend(loc="upper center", prop={"size": 10}, ncol=4)
            hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=138, ax=ax)

            self.figures.append((fig, ax, output_name))


class XSNLLsPerCategory(Figure):
    """ Plot the NLLs for a given category, one NLL per POI
    """

    def __init__(self, differential_spectrum):
        self.ds = differential_spectrum
        self.fig, self.ax = plt.subplots()
        self.output_name = "NLLs_{}_{}".format(self.ds.variable, self.ds.category)

        # Set labels
        self.ax.set_xlabel(self.ds.variable)
        self.ax.set_ylabel("-2$\Delta$lnL")

        # Set limits
        self.ax.set_ylim(0.0, 8.0)

        # Draw horizontal line at 1
        self.ax.axhline(1.0, color="k", linestyle="--")

        # Draw all the NLLs on the ax
        logger.debug(differential_spectrum.scans)
        rainbow_iter = cycle(rainbow)
        for poi_scan in differential_spectrum.scans.items():
            color = next(rainbow_iter)
            poi, scan = poi_scan
            self.ax = scan.plot(self.ax, color)

        # Legend
        self.ax.legend(loc="upper center", prop={"size": 10}, ncol=4)
        hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=138, ax=self.ax)


class DiffXSsPerObservable(Figure):
    """
    """

    def __init__(
        self, output_name, sm_shape, observable_shapes, observable_shapes_systonly=None
    ):
        if observable_shapes_systonly is None:
            observable_shapes_systonly = []
        self.output_name = output_name
        # Set up figure and axes
        self.fig, (self.main_ax, self.ratio_ax) = plt.subplots(
            nrows=2, ncols=1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True
        )

        # X limits depend on the SM plot (the one in the background with the predictions)
        logger.debug(f"Using SM shape:\n{sm_shape}")
        self.main_ax.set_xlim(sm_shape.fake_edges[0], sm_shape.fake_edges[-1])
        self.ratio_ax.set_xlim(sm_shape.fake_edges[0], sm_shape.fake_edges[-1])
        self.main_ax.set_yscale("log")
        self.ratio_ax.set_ylim(0, 2)
        self.ratio_ax.set_yticks([0, 1, 2])
        self.main_ax, self.ratio_ax = sm_shape.plot(self.main_ax, self.ratio_ax)
        # in the case of Njets, labels are in the middle of the bins
        if sm_shape.observable in ["Njets"]:
            width = sm_shape.edges[1] - sm_shape.edges[0]
            start = sm_shape.edges[0] + width / 2
            logger.debug(f"Start: {start}")
            end = sm_shape.edges[-1] + width / 2
            logger.debug(f"End: {end}")
            logger.debug(f"Setting x ticks to {np.arange(start, end, width)}")
            self.ratio_ax.set_xticks(np.arange(start, end, width), minor=True)
            logger.debug(f"Minor ticks: {self.ratio_ax.get_xticks(minor=True)}")
            tick_labels = deepcopy(sm_shape.edges[:-1])
            tick_labels[-1] = f"{tick_labels[-1]}+"
            logger.debug(f"Setting x tick labels to {tick_labels}")
            self.ratio_ax.set_xticklabels("")
            self.ratio_ax.set_xticklabels(tick_labels, minor=True)
        else:
            self.ratio_ax.set_xticks(sm_shape.fake_edges)
            tick_labels = deepcopy(sm_shape.edges)
            if sm_shape.overflow:
                tick_labels[-1] = r"$\infty$"
            self.ratio_ax.set_xticklabels(tick_labels)
        self.ratio_ax.tick_params(axis="x", which="major", labelsize=13)
        self.main_ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        self.ratio_ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        self.ratio_ax.set_xlabel(observable_specs[sm_shape.observable]["x_plot_label"])
        self.main_ax.set_ylabel(observable_specs[sm_shape.observable]["y_plot_label"])
        self.ratio_ax.set_ylabel("Ratio to prediction", size=15)
        self.ratio_ax.grid(which="both", axis="y")
        self.main_ax.grid(which="major", axis="x", linestyle="-", alpha=0.3)
        self.ratio_ax.grid(which="major", axis="x", linestyle="-", alpha=0.3)

        # Horrible way to somehow move the points away from each other and not have them superimposed
        # displacements = [0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6]
        # categories = [shape.category for shape in observable_shapes]
        # displacements_dict = {k: v for k, v in zip(categories, displacements)}
        displacements_dict = {
            "smH_PTH": {"HggHZZ": 0, "HggHZZHWW": 0, "Hgg": 0.2, "HZZ": -0.2, "HWW": 0},
            "Njets": {"HggHWW": 0, "Hgg": 0.2, "HWW": -0.2},
            "yH": {"HggHZZ": 0, "HggHZZHWW": 0, "Hgg": 0.2, "HZZ": -0.2, "HWW": 0},
        }
        logger.debug(f"Displacements: {displacements_dict}")

        for shape in observable_shapes:
            shape.fake_rebin(sm_shape)

            displacement = displacements_dict[sm_shape.observable][shape.category]
            shape.fake_maybe_moved_centers = (
                shape.fake_centers + np.diff(shape.fake_edges) * displacement
            )

            # Add dashed horizontal line only in cases in which shape and SM shape edges differ
            # The line has to be added only for the bins that actually differ, not in all of them
            # The procedure is thus a bit convoluted
            bins_with_hline = []
            logger.debug("Looking for bins with hline")
            logger.debug(f"SM shape edges: {sm_shape.edges}")
            logger.debug(f"Shape edges: {shape.edges}")
            for i, edges_pair in enumerate(zip(shape.edges, shape.edges[1:])):
                left_edge = edges_pair[0]
                right_edge = edges_pair[1]
                # get index of these edges in the SM shape
                sm_left_edge_index = sm_shape.edges.index(left_edge)
                sm_right_edge_index = sm_shape.edges.index(right_edge)
                if sm_right_edge_index - sm_left_edge_index > 1:
                    bins_with_hline.append(i)
            logger.debug(f"Bins with hline: {bins_with_hline}")

            # Prediction shape for ratio plot
            prediction = deepcopy(sm_shape)
            prediction.rebin(shape.edges)
            self.main_ax, self.ratio_ax = shape.plot(
                self.main_ax, self.ratio_ax, prediction, bins_with_hline
            )

        # Add bands for systonly
        for shape_systonly in observable_shapes_systonly:
            logger.debug("Adding bands for systonly")
            shape_systonly.fake_rebin(sm_shape)

            displacement = displacements_dict[sm_shape.observable][
                shape_systonly.category
            ]
            shape_systonly.fake_maybe_moved_centers = (
                shape_systonly.fake_centers
                + np.diff(shape_systonly.fake_edges) * displacement
            )

            prediction = deepcopy(sm_shape)
            prediction.rebin(shape_systonly.edges)
            self.main_ax, self.ratio_ax = shape_systonly.plot_as_band(
                self.main_ax, self.ratio_ax, prediction
            )

        # Miscellanea business that has to be done after
        self.main_ax.legend(loc="lower left")

        hep.cms.label(
            loc=0, data=True, llabel="Work in Progress", lumi=138, ax=self.main_ax
        )

