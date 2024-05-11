from matplotlib import figure
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

hep.style.use("CMS")
from itertools import cycle
from copy import deepcopy

from .cosmetics import (
    rainbow,
    observable_specs,
    category_specs,
    fit_type_colors,
    TK_parameters_labels,
    SMEFT_parameters_labels,
)

bsm_parameters_labels = {}
for k, v in TK_parameters_labels.items():
    bsm_parameters_labels[k] = v
for k, v in SMEFT_parameters_labels.items():
    bsm_parameters_labels[k] = v
from .shapes import (
    ObservableShapeSM,
    smH_PTH_EvenMoreMaximumGranularity_obs_shape,
    yH_Granular_obs_shape,
)

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
        logger.debug(f"Saved {self.output_name} to {output_dir} in pdf and png")


class GenericNLLsPerPOI(Figure):
    """
    Introduced to plot following NLLS in SMEFT scans:
    fig1:
        - coeff: chg
        - observed
        - NLLS: Hgg HZZ Comb
    But it can probably be used in other contexts

    scans is a dictionary like the folliwing:
    {
        "Hgg": Scan,
        "HZZ": Scan,
    }
    """

    def __init__(
        self,
        poi,
        scans,
        subcategory,
        simple=False,
        full_range=False,
        plot_string=False,
        plot_interval=False,
        minimum_vertical_line=True,
        logy=False,
    ):
        self.scans = scans
        self.categories = list(scans.keys())
        categories_string = "-".join(self.categories)
        self.fig, self.ax = plt.subplots()
        self.output_name = f"NLLs_{poi}_{categories_string}_{subcategory}"
        if full_range:
            self.output_name = (
                f"NLLs_{poi}_{categories_string}_{subcategory}_full_range"
            )

        # Set labels
        try:
            self.ax.set_xlabel(bsm_parameters_labels[poi])
        except KeyError:
            self.ax.set_xlabel(poi)
        self.ax.set_ylabel("-2$\Delta$lnL")

        # Set limits
        if not full_range and not logy:
            self.ax.set_ylim(0.0, 8.0)

        # log scale
        if logy:
            self.ax.set_yscale("log")

        # Draw horizontal line at 1 and 4
        if not full_range:
            self.ax.axhline(1.0, color="k", linestyle="--")
            self.ax.axhline(3.84, color="k", linestyle="--")

        start_y_for_text = 0.6
        for scan_name, scan in scans.items():
            try:
                color = category_specs[scan_name]["color"]
                label = category_specs[scan_name]["plot_label"]
            except KeyError:
                color = "purple"
                label = scan_name
            if simple:
                self.ax = scan.plot_simple(
                    self.ax,
                    color=color,
                    label=label,
                    ylim=1000000 if full_range else 8.0,
                )
            else:
                self.ax = scan.plot(
                    self.ax,
                    color=color,
                    label=label,
                    ylim=1000000 if full_range else 8.0,
                    minimum_vertical_line=minimum_vertical_line
                )
            if plot_string:
                # also add text with best fit
                best_fit_string = scan.get_bestfit_string()
                self.ax.text(
                    0.5,
                    start_y_for_text,
                    best_fit_string,
                    color=color,
                    fontsize=20,
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                )
                start_y_for_text -= 0.05
            if plot_interval:
                # also add text with best fit
                best_fit_string = scan.get_95interval_string()
                self.ax.text(
                    0.5,
                    start_y_for_text,
                    best_fit_string,
                    color=color,
                    fontsize=20,
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                )
                start_y_for_text -= 0.05

        # Legend
        self.ax.legend(loc="upper center", prop={"size": 16}, ncol=4)
        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.ax)


class TwoScans(Figure):
    def __init__(
        self,
        poi,
        category,
        scan1,
        scan2,
        label1,
        label2,
        prefix="TwoScans",
        bestfit1=True,
        bestfit2=True,
    ):
        self.fig, self.ax = plt.subplots()
        self.output_name = f"{prefix}_{poi}_{category}"

        # Set labels
        try:
            self.ax.set_xlabel(bsm_parameters_labels[poi])
        except KeyError:
            self.ax.set_xlabel(poi)
        self.ax.set_ylabel("-2$\Delta$lnL")

        # Set limits
        self.ax.set_ylim(0.0, 8.0)

        # Draw horizontal line at 1 and 4
        self.ax.axhline(1.0, color="k", linestyle="--")
        self.ax.axhline(3.84, color="k", linestyle="--")

        self.ax = scan1.plot_simple(self.ax, "k", label=label1, ylim=8.0)
        self.ax = scan2.plot_simple(
            self.ax, "red", label=label2, ylim=8.0, linestyle="--"
        )
        # see https://stackoverflow.com/questions/8482588/putting-text-in-top-left-corner-of-matplotlib-plot for text coordinates
        if bestfit1:
            best_fit_string1 = scan1.get_bestfit_string()
            self.ax.text(
                0.5,
                0.75,
                best_fit_string1,
                color="k",
                fontsize=14,
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
        if bestfit2:
            best_fit_string2 = scan2.get_bestfit_string()
            self.ax.text(
                0.5,
                0.65,
                best_fit_string2,
                color="red",
                fontsize=14,
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )

        # Legend
        self.ax.legend(loc="upper center", prop={"size": 16}, ncol=4)
        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.ax)


class ScanChiSquare(Figure):
    def __init__(self, poi, scan, chi_dct):
        self.fig, self.ax = plt.subplots()
        self.output_name = f"ScanChiSquare_{poi}"

        # Set labels
        try:
            self.ax.set_xlabel(bsm_parameters_labels[poi])
        except KeyError:
            self.ax.set_xlabel(poi)
        self.ax.set_ylabel("-2$\Delta$lnL")

        # Set limits
        self.ax.set_ylim(0.0, 8.0)

        # Draw horizontal line at 1 and 4
        self.ax.axhline(1.0, color="k", linestyle="--")
        self.ax.axhline(3.84, color="k", linestyle="--")

        self.ax = scan.plot_simple(self.ax, "k", label="Scan", ylim=8.0)

        # second axis
        self.ax2 = self.ax.twinx()

        # Set labels
        self.ax2.set_ylabel("$\Delta\chi^2$")

        # Set limits
        self.ax2.set_ylim(0.0, 8.0)

        # plot
        self.ax2.plot(
            chi_dct["values"], chi_dct["chi_square"], color="red", label="$\chi^2$"
        )

        # Legend
        self.ax.legend(loc="upper left", prop={"size": 16}, ncol=4)
        self.ax2.legend(loc="upper right", prop={"size": 16}, ncol=4)
        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.ax)


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
            ax.axhline(3.84, color="k", linestyle="--")

            # Draw all the NLLs with different colors
            for scan_tpl, color in zip(scans, fit_type_colors):
                ax = scan_tpl[1].plot(ax, color, label=scan_tpl[0])
                if debug:
                    ax = scan_tpl[1].plot_original_points(
                        ax, color, label=f"{scan_tpl[0]} (original)"
                    )

            # Add note with minimum and uncertainties
            nomstring = f"{nominal_spectrum.scans[poi].minimum[0]:.3f}"
            upstring = f"{nominal_spectrum.scans[poi].up68_unc[0]:.3f}"
            upstring = "{+" + upstring + "}"
            downstring = f"{nominal_spectrum.scans[poi].down68_unc[0]:.3f}"
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
            hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=ax)

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
            hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=ax)

            self.figures.append((fig, ax, output_name))


class XSNLLsPerCategory(Figure):
    """Plot the NLLs for a given category, one NLL per POI"""

    def __init__(self, differential_spectrum, ylim=8.0, print_best=False):
        self.ds = differential_spectrum
        self.fig, self.ax = plt.subplots()
        self.output_name = "NLLs_{}_{}".format(self.ds.variable, self.ds.category)

        # Set labels
        self.ax.set_xlabel(self.ds.variable)
        self.ax.set_ylabel("-2$\Delta$lnL")

        # Set limits
        self.ax.set_ylim(0.0, ylim)

        # Draw horizontal line at 1 and 4
        self.ax.axhline(1.0, color="k", linestyle="--")
        self.ax.axhline(3.84, color="k", linestyle="--")

        # Draw all the NLLs on the ax
        logger.debug(differential_spectrum.scans)
        rainbow_iter = cycle(rainbow)
        for poi_scan in differential_spectrum.scans.items():
            color = next(rainbow_iter)
            poi, scan = poi_scan
            self.ax = scan.plot(self.ax, color, ylim=ylim)

        # Used just for the case of quick_scan
        if print_best:
            nomstring = f"{differential_spectrum.scans[poi].minimum[0]:.5f}"
            upstring = f"{differential_spectrum.scans[poi].up68_unc[0]:.5f}"
            upstring = "{+" + upstring + "}"
            downstring = f"{differential_spectrum.scans[poi].down68_unc[0]:.5f}"
            downstring = "{-" + downstring + "}"
            self.ax.text(
                differential_spectrum.scans[poi].minimum[0],
                2.5,
                f"{poi} = ${nomstring}^{upstring}_{downstring}$",
                fontsize=14,
                ha="center",
                va="center",
            )

        # Legend
        self.ax.legend(loc="upper center", prop={"size": 10}, ncol=4)
        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.ax)


class DiffXSsPerObservable(Figure):
    """ """

    def __init__(
        self,
        output_name,
        sm_shape,
        observable_shapes,
        observable_shapes_systonly=None,
        other_sm_shapes_dicts=None,
        kappa_prediction=None
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
        large_ratio = False
        for os in observable_shapes:
            if os.category in [
                "HggHZZHWWHttHbbVBF",
                "HggHZZHWWHttHbbVBFHttBoost",
                "HbbVBF",
            ]:
                large_ratio = True
        if large_ratio:
            # self.ratio_ax.set_ylim(-6, 6)
            # self.ratio_ax.set_yticks([-6, -3, 0, 3, 6])
            self.ratio_ax.set_ylim(0, 2)
            self.ratio_ax.set_yticks([0, 1, 2])
        else:
            self.ratio_ax.set_ylim(0, 2)
            self.ratio_ax.set_yticks([0, 1, 2])
        if "Njets" in output_name:
            self.ratio_ax.set_ylim(0, 2)
            self.ratio_ax.set_yticks([0, 1, 2])
        if "DEtajj" in output_name:
            self.ratio_ax.set_ylim(-1, 2)
            self.ratio_ax.set_yticks([-1, 0, 1, 2])
        if "mjj" in output_name:
            self.ratio_ax.set_ylim(-1, 3)
            self.ratio_ax.set_yticks([-1, 0, 1, 2, 3])
        if "smH_PTJ0" in output_name:
            self.ratio_ax.set_ylim(-1, 3)
            self.ratio_ax.set_yticks([-1, 0, 1, 2, 3])

        self.main_ax, self.ratio_ax = sm_shape.plot(self.main_ax, self.ratio_ax)

        # other_sm_shapes has to be a list of dictionaries
        # {"shape": value, "color": value, "label": value}
        if other_sm_shapes_dicts is not None:
            for other_sm_shape_dct in other_sm_shapes_dicts:
                self.main_ax, self.ratio_ax = other_sm_shape_dct["shape"].plot_other(
                    self.main_ax,
                    self.ratio_ax,
                    color=other_sm_shape_dct["color"],
                    label=other_sm_shape_dct["label"],
                    other=sm_shape,
                    where=other_sm_shape_dct["where"],
                )
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
            self.ratio_ax.set_xticklabels(
                tick_labels, rotation=45 if sm_shape.observable == "smH_PTH" else 0
            )
        self.ratio_ax.tick_params(axis="x", which="major", labelsize=13)
        self.main_ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        self.ratio_ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        self.ratio_ax.set_xlabel(observable_specs[sm_shape.observable]["x_plot_label"])
        self.main_ax.set_ylabel(observable_specs[sm_shape.observable]["y_plot_label"])
        self.ratio_ax.set_ylabel("Ratio to prediction", size=15)
        self.ratio_ax.grid(which="both", axis="y")
        self.main_ax.grid(which="major", axis="x", linestyle="-", alpha=0.3)
        self.ratio_ax.grid(which="major", axis="x", linestyle="-", alpha=0.3)

        # if kappa_prediction is there, fake_rebin it based on sm_shape and plot it
        if kappa_prediction is not None:
            kappa_prediction.fake_rebin(sm_shape)
            self.main_ax, self.ratio_ax = kappa_prediction.plot(self.main_ax, self.ratio_ax)

        # Horrible way to somehow move the points away from each other and not have them superimposed
        # displacements = [0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6]
        # categories = [shape.category for shape in observable_shapes]
        # displacements_dict = {k: v for k, v in zip(categories, displacements)}
        displacements_dict = {
            "smH_PTH": {
                "HggHZZ": 0,
                "HggHZZHWW": 0,
                "HggHZZHWWHtt": 0,
                "HggHZZHWWHttHbb": 0,
                "HggHZZHWWHttHbbVBF": 0,
                "HggHZZHWWHttHbbVBFHttBoost": 0,
                "FinalComb": 0,
                "Hgg": 0.2,
                "HZZ": -0.2,
                "HWW": 0,
                "Htt": -0.4,
                "Hbb": -0.2,
                "HbbVBF": 0.2,
                "HttBoost": -0.2,
            },
            "Njets": {
                "HggHWW": 0,
                "HggHWWHtt": 0,
                "HggHZZHWWHtt": 0,
                "Hgg": 0.2,
                "HZZ": -0.2,
                "HWW": 0.4,
                "Htt": -0.4,
            },
            "yH": {"HggHZZ": 0, "HggHZZHWW": 0, "Hgg": 0.2, "HZZ": -0.2, "HWW": 0},
            "smH_PTJ0": {"HggHZZHttBoost": 0, "FinalComb": 0, "Hgg": 0.2, "HZZ": -0.2, "HttBoost": 0.2},
            "mjj": {"HggHZZ": 0, "Hgg": 0.2, "HZZ": -0.2},
            "DEtajj": {"HggHZZ": 0, "Hgg": 0.2, "HZZ": -0.2},
            "TauCJ": {"HggHZZ": 0, "Hgg": 0.2, "HZZ": -0.2},
        }
        logger.debug(f"Displacements: {displacements_dict}")

        passed_sm_shape = sm_shape  # che brutta roba che mi tocca fare
        for shape in observable_shapes:
            # Fuckin Hbb porcodiqueldio
            if (
                shape.observable == "smH_PTH"
                and shape.category.startswith("Hbb")
                and shape.category != "HbbVBF"
            ):
                sm_shape = smH_PTH_EvenMoreMaximumGranularity_obs_shape
                shape.fake_edges = np.array([16.5, 18, 19, 20])
                shape.fake_centers = np.array([17.25, 18.5, 19.5])
                shape.fake_maybe_moved_centers = np.array([17.25, 18.5, 19.5])
                logger.debug(f"Mannaggia a Hbb fake edges: {shape.fake_edges}")
            else:
                sm_shape = passed_sm_shape
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
        sm_shape = passed_sm_shape

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
        location = "lower left"
        prop = {"size": 14}
        if sm_shape.observable == "Njets":
            location = "upper right"
            prop = {"size": 12}
        self.main_ax.legend(loc=location, prop=prop)

        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.main_ax)

        
class EasyTwoDScan(Figure):
    def __init__(
        self,
        scan_dict,
        output_name,
    ):
        super().__init__()
        self.scan_dict = scan_dict
        self.output_name = output_name
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(18, 14))
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        
        pois = list(scan_dict.values())[0].pois

        for (name, scan), color in zip(scan_dict.items(), colors):
            self.ax = scan.plot_as_contour(self.ax, color=color, label=name)

        self.ax.set_xlabel(bsm_parameters_labels[pois[0]])
        self.ax.set_ylabel(bsm_parameters_labels[pois[1]]) 
        self.ax.legend(loc="upper left")
        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.ax)


class TwoDScansPerModel(Figure):
    def __init__(
        self,
        scan_dict,
        combination_name,
        model_config,
        combination_asimov_scan=None,
        output_name=None,
        is_asimov=False,
        force_limit=False,
        legend_conf={"loc": "upper left", "prop": {"size": 14}}
    ):
        """
        scan_dict is e.g. {"Hgg": Scan2D}
        combination_name: which one has to be plotted as combination
        model_config: a dictionary of the form {"kappac": (-8, 8), "kappab": (-2, 2)}
        combination_asimov_scan: for the case in which the background has to be expected,
        use this scan instead of the combination_name one
        force_limit: if True, the limit on x and y is forced to be the one from model_config (instead of the max between model_config and the scan)
        """
        super().__init__()
        self.scan_dict = scan_dict

        self.fig, self.ax = plt.subplots(1, 1, figsize=(18, 14))
        # Plot the combination one
        if combination_asimov_scan is not None:
            self.ax, self.colormap, self.pc = combination_asimov_scan.plot_as_heatmap(
                self.ax
            )
            # this is because we assign an attribute category to the scan only in the SMEFT case!! (for now)
            try:
                self.ax = combination_asimov_scan.plot_as_contour(
                    self.ax,
                    color="yellow",
                    label=f"Exp. {category_specs[combination_asimov_scan.category]['plot_label']}",
                )
            except:
                self.ax = combination_asimov_scan.plot_as_contour(
                    self.ax, color="yellow", label=f"Exp. Combination"
                )
            # self.ax = combination_asimov_scan.plot_as_contourf(self.ax)
        else:
            self.ax, self.colormap, self.pc = self.scan_dict[
                combination_name
            ].plot_as_heatmap(self.ax)
            # self.ax = self.scan_dict[combination_name].plot_as_contourf(self.ax)
        self.fig.colorbar(self.pc, ax=self.ax, label="-2$\Delta$lnL")

        # Combination + others as countour
        for category, scan in scan_dict.items():
            try:
                self.ax = self.scan_dict[category].plot_as_contour(
                    self.ax,
                    color=category_specs[category.split("_")[0]]["color"],
                    label=f'{"Exp. " if is_asimov else ""}{category_specs[category.split("_")[0]]["plot_label"]}',
                )
            except KeyError:  # quick_scan case, in which the name is "test"
                self.ax = self.scan_dict[category].plot_as_contour(
                    self.ax, color="black"
                )

        if output_name is not None:
            self.output_name = output_name

        # set limits on x and y
        poi1, poi2 = list(model_config.keys())
        x_left = np.max(
            [model_config[poi1][0], *[np.min(s.x_int) for s in scan_dict.values()]]
        )
        x_right = np.min(
            [model_config[poi1][1], *[np.max(s.x_int) for s in scan_dict.values()]]
        )
        y_down = np.max(
            [model_config[poi2][0], *[np.min(s.y_int) for s in scan_dict.values()]]
        )
        y_up = np.min(
            [model_config[poi2][1], *[np.max(s.y_int) for s in scan_dict.values()]]
        )
        if force_limit:
            x_left = model_config[poi1][0]
            x_right = model_config[poi1][1]
            y_down = model_config[poi2][0]
            y_up = model_config[poi2][1]
        self.ax.set_xlim(x_left, x_right)
        self.ax.set_ylim(y_down, y_up)
        # Miscellanea business that has to be done after
        try:
            self.ax.set_xlabel(
                bsm_parameters_labels[scan_dict[combination_name].pois[0]]
            )
            self.ax.set_ylabel(
                bsm_parameters_labels[scan_dict[combination_name].pois[1]]
            )
        except KeyError:
            try:
                self.ax.set_xlabel(bsm_parameters_labels[combination_asimov_scan.pois[0]])
                self.ax.set_ylabel(bsm_parameters_labels[combination_asimov_scan.pois[1]])
            except AttributeError:
                self.ax.set_xlabel(poi1)
                self.ax.set_ylabel(poi2)
        # self.colormap.ax.set_ylabel("-2$\Delta$lnL")
        self.ax.legend(**legend_conf)

        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.ax)


class TwoDScanDebug(Figure):
    def __init__(self, scan, output_name=None):
        self.output_name = output_name
        x, y, z = scan.points
        self.fig, self.ax = plt.subplots(1, 1, figsize=(18, 14))
        x_mesh, y_mesh = np.meshgrid(x, y)
        self.ax.pcolormesh(x_mesh, y_mesh, np.tile(z, (len(y), 1)), cmap="viridis")
        self.ax.set_xlabel(scan.pois[0])
        self.ax.set_ylabel(scan.pois[1])


# Used for quick_scan only
class NScans(Figure):
    def __init__(self, scan_dict, ylim=8.0):
        """
        scan_dict is e.g. {"Hgg": Scan1D}
        """
        self.fig, self.ax = plt.subplots()
        self.output_name = "NLLs_" + "_".join(scan_dict.keys())

        # Set labels
        self.ax.set_xlabel(list(scan_dict.values())[0].poi)
        self.ax.set_ylabel("-2$\Delta$lnL")

        # Set limits
        self.ax.set_ylim(0.0, ylim)

        # Draw all the NLLs on the ax
        for scan_name, scan in scan_dict.items():
            self.ax = scan.plot(self.ax, label=scan_name)

        # Draw horizontal line at 1 and 4
        self.ax.axhline(1.0, color="k", linestyle="--")
        self.ax.axhline(3.84, color="k", linestyle="--")

        # Legend
        self.ax.legend(loc="upper center", prop={"size": 10}, ncol=4)
        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.ax)


class SMEFTSummaryPlot(Figure):
    def __init__(self, wc_scans):
        """
        Plot a summary of the SMEFT results.
        wc_scans is e.g.: {"chg": {"PtFullComb": Scan1D, "PtFullComb_asimov": Scan1D}}
        """
        self.fig, self.ax = plt.subplots(
            figsize=(10, 20)
        )
        # get first key of the dictionary
        fk = list(wc_scans.keys())[0]
        categories = list(wc_scans[fk].keys())
        
        self.output_name = "SMEFT_summary_" + "_".join(wc_scans.keys()) + "_{}".format("-".join(categories))

        # vertical line at 0
        self.ax.axvline(0.0, color="k", linestyle="-")

        # assign each WC to an order of magnitude
        wc_scans_first = {wc: wc_scans[wc][categories[0]] for wc in wc_scans}
        wc_orders = {}
        for wc, scan in wc_scans_first.items():
            rng = np.abs(scan.up95_unc)
            wc_orders[wc] = np.floor(np.log10(rng))
        
        # length of vertical axis will be the number of WCs + 2 (in order to have some space at the top and bottom)
        y_min = 0
        y_max = len(wc_scans) + 1
        
        # tick labels will be the WCs
        tick_labels = []

        colors = ["black", "red"]
        for y_pos_diff, (wc, order) in enumerate(wc_orders.items()):
            col_index = 0
            for cat_displace, cat in enumerate(categories):
                try:
                    cat_label = category_specs[cat]["plot_label"]
                except KeyError:
                    cat_label = cat
                cat_displace = cat_displace * 0.1
                color = colors[col_index]
                y_pos = y_max - y_pos_diff - 1 + cat_displace
                scan = wc_scans[wc][cat]
                denominator = 10 ** order
                centre = scan.minimum[0] / denominator
                extrema_95 = scan.down95[0][0] / denominator, scan.up95[0][0] / denominator
                extrema_68 = scan.down68[0][0] / denominator, scan.up68[0][0] / denominator

                self.ax.plot(
                    extrema_95,
                    [y_pos, y_pos],
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    label=f"{cat_label} 95% CL" if y_pos_diff == 0 else "",
                )
                self.ax.plot(
                    extrema_68,
                    [y_pos, y_pos],
                    color=color,
                    linestyle="-",
                    linewidth=3,
                    label=f"{cat_label} 68% CL" if y_pos_diff == 0 else "",
                )
                self.ax.plot(
                    centre,
                    y_pos,
                    color=color,
                    marker="o",
                    markersize=10,
                    linestyle="",
                    label=f"{cat_label} Best fit" if y_pos_diff == 0 else "",
                )

                label = bsm_parameters_labels[wc]
                if order not in [0, 1]:
                    label += " " + r"$\times$" + f"$10^{{{int(order)}}}$"
                elif order == 1:
                    label += " " + r"$\times$" + f"$10$"
                if cat_displace == 0:
                    tick_labels.append(label)
                col_index += 1

        # set limits on y
        self.ax.set_ylim(y_min, y_max)
        
        # revert the order of the tick labels and set them
        tick_labels = tick_labels[::-1]
        self.ax.set_yticks(np.arange(y_min + 1, y_max))
        self.ax.set_yticklabels(tick_labels, fontsize=20)
        # now that we put the labels, do not show the ticks
        self.ax.tick_params(axis="y", which="both", length=0)
        
        # set limits on x
        x_min = -15
        x_max = 15
        self.ax.set_xlim(x_min, x_max)
        # set x label
        self.ax.set_xlabel("Parameter value", fontsize=20, horizontalalignment="center")
        
        hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=self.ax)

        # legend
        self.ax.legend(loc="upper left", prop={"size": 12}, ncol=1)