from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import cycle
from copy import deepcopy

from .physics import (
    smH_PTH_Hgg_xs,
    smH_PTH_MaximumGranularity_xs,
    smH_PTH_EvenMoreMaximumGranularity_xs,
    Njets_Hgg_xs,
    yH_Hgg_xs,
    yH_Granular_xs,
    smH_PTJ0_Hgg_xs,
    smH_PTJ0_Granular_xs,
    mjj_Hgg_xs,
    DEtajj_Hgg_xs,
    TauCJ_Hgg_xs,
    smH_PTH_Hgg_xs_noNNLOPS,
    smH_PTH_EvenMoreMaximumGranularity_xs_noNNLOPS,
    Njets_Hgg_xs_noNNLOPS,
    yH_Granular_xs_noNNLOPS,
    smH_PTJ0_Hgg_xs_noNNLOPS,
    smH_PTJ0_Granular_xs_noNNLOPS,
    mjj_Hgg_xs_noNNLOPS,
    DEtajj_Hgg_xs_noNNLOPS,
    TauCJ_Hgg_xs_noNNLOPS,
    smH_PTH_Hgg_xs_powheg, 
    smH_PTH_EvenMoreMaximumGranularity_xs_powheg,
    Njets_Hgg_xs_powheg,
    yH_Granular_xs_powheg,
    smH_PTJ0_Hgg_xs_powheg,
    smH_PTJ0_Granular_xs_powheg,
    mjj_Hgg_xs_powheg,
    DEtajj_Hgg_xs_powheg,
    TauCJ_Hgg_xs_powheg,
    analyses_edges,
)
from .cosmetics import markers, category_specs, TK_parameters_labels

import logging

logger = logging.getLogger(__name__)


class ObservableShape:
    """
    """

    def __init__(
        self,
        observable,
        category,
        edges,
        nominal_values,
        up_values,
        down_values,
        overflow=True,
    ):

        self.observable = observable
        self.category = category
        self.edges = edges
        self.xs = nominal_values
        self.xs_up = up_values
        self.xs_down = down_values
        self.overflow = overflow

        # Since we want every bin of the same width, we can do this for the steps
        self.fake_bin_width = 1
        self.fake_edges = np.arange(0, len(self.edges), self.fake_bin_width)
        self.fake_centers = (self.fake_edges[1:] + self.fake_edges[:-1]) / 2
        self.fake_maybe_moved_centers = self.fake_centers.copy()

    @property
    def nbins(self):
        return len(self.xs)

    @property
    def inclusive_xs(self):
        return np.sum(self.xs)

    @property
    def bin_widths(self):
        bin_width = np.diff(self.edges)
        if self.overflow:
            bin_width[-1] = bin_width[-2]
        return bin_width

    @property
    def xs_over_bin_width(self):
        return self.xs / self.bin_widths

    @property
    def xs_up_over_bin_width(self):
        return self.xs_up / self.bin_widths

    @property
    def xs_down_over_bin_width(self):
        return self.xs_down / self.bin_widths

    def rebin(self, new_edges):
        self.xs = self.merge_bins(self.xs, self.edges, new_edges)
        self.xs_up = self.merge_bins(self.xs_up, self.edges, new_edges)
        self.xs_down = self.merge_bins(self.xs_down, self.edges, new_edges)
        self.edges = new_edges
        self.fake_edges = np.arange(0, len(self.edges), self.fake_bin_width)
        self.fake_centers = (self.fake_edges[1:] + self.fake_edges[:-1]) / 2
        self.fake_maybe_moved_centers = self.fake_centers.copy()

    def fake_rebin(self, other_shape):
        logger.debug(f"Stretching fake range for {self.observable} {self.category}")
        logger.debug(f"Current Fake edges: {self.fake_edges}")
        logger.debug(f"Other Fake edges: {other_shape.fake_edges}")
        self.fake_edges = []
        self.fake_centers = []
        new_unit = other_shape.fake_bin_width
        new_edge = other_shape.edges.index(self.edges[0])
        # new_edge = other_shape.fake_edges[0]
        self.fake_edges.append(new_edge)
        for first, second in zip(self.edges, self.edges[1:]):
            other_first_index = other_shape.edges.index(first)
            other_second_index = other_shape.edges.index(second)
            how_many_bins_between = other_second_index - other_first_index
            extension = how_many_bins_between * new_unit
            new_edge += extension
            self.fake_edges.append(new_edge)
        self.fake_edges = np.array(self.fake_edges)
        self.fake_centers = (self.fake_edges[1:] + self.fake_edges[:-1]) / 2
        self.fake_maybe_moved_centers = self.fake_centers.copy()
        logger.debug(f"New Fake edges after fake_rebin: {self.fake_edges}")

    def merge_bins(self, old_bins, old_edges, new_edges):
        if not all(edge in old_edges for edge in new_edges):
            print("Old edges: {}".format(old_edges))
            print("New edges: {}".format(new_edges))
            raise ValueError("New edges are not a subset of old edges")

        # Loop over pairs of consecutive edges in new_edges
        new_bins = []
        for first, second in zip(new_edges, new_edges[1:]):
            old_first_index = old_edges.index(first)
            old_second_index = old_edges.index(second)
            new_bins.append(sum(old_bins[old_first_index:old_second_index]))

        if len(new_bins) != len(new_edges) - 1:
            raise ValueError("Something went wrong")

        return np.array(new_bins)

    def __str__(self):
        string = f"Observable: {self.observable}\n"
        string += f"Edges: {self.edges}\n"
        string += f"Nominal XS: {self.xs}\n"
        string += f"Up XS: {self.xs_up}\n"
        string += f"Down XS: {self.xs_down}\n"
        string += f"Nominal XS / bin width: {self.xs_over_bin_width}\n"
        string += f"Up XS / bin width: {self.xs_up_over_bin_width}\n"
        string += f"Down XS / bin width: {self.xs_down_over_bin_width}\n"

        return string


class ObservableShapeSM(ObservableShape):
    """

    Remember:
                +------------------+
                |                  |
              height               |
                |                  |
               (xy)---- width -----+
    """

    def plot(self, ax, rax):
        ax.stairs(
            edges=self.fake_edges,
            values=self.xs_over_bin_width,
            color="grey",
            linewidth=1,
            label="aMC@NLO, NNLOPS",
        )

        rax.axhline(y=1, color="grey", linewidth=1)

        # Apply rectangular patches for uncertainties
        xs_up_fraction = self.xs_up / self.xs
        xs_down_fraction = self.xs_down / self.xs
        for x, xs_obw, xs_up_obw, xs_down_obw, xs_up_frac, xs_down_frac in zip(
            self.fake_edges[:-1],
            self.xs_over_bin_width,
            self.xs_up_over_bin_width,
            self.xs_down_over_bin_width,
            xs_up_fraction,
            xs_down_fraction,
        ):
            # Main plot
            ax.add_patch(
                Rectangle(
                    xy=(x, xs_down_obw),
                    width=self.fake_bin_width,
                    height=xs_up_obw - xs_down_obw,
                    hatch="/////",
                    fill=False,
                    color="grey",
                    linewidth=0,
                )
            )
            # Ratio plot
            rax.add_patch(
                Rectangle(
                    xy=(x, xs_down_frac),
                    width=1,
                    height=xs_up_frac - xs_down_frac,
                    hatch="/////",
                    fill=False,
                    color="grey",
                    linewidth=0,
                )
            )

        return ax, rax

    def plot_other(self, ax, rax, color, label, other, where="left"):
        # plot horizontal lines like in plot function but not stairs
        ax.hlines(
            self.xs_over_bin_width,
            self.fake_edges[:-1],
            self.fake_edges[1:],
            color=color,
            linewidth=1,
            label=label,
        )

        rax.hlines(
            self.xs_over_bin_width / other.xs_over_bin_width,
            self.fake_edges[:-1],
            self.fake_edges[1:],
            color=color,
            linewidth=1,
        )

        # patches for uncertainties
        # if right is chosen, the patch is on the right side of the bin
        # if left is chosen, the patch is on the left side of the bin
        xs_up_fraction = self.xs_up / other.xs
        xs_down_fraction = self.xs_down / other.xs
        if where == "left":
            offset = 0
        elif where == "right":
            offset = 0.5

        for x, xs_obw, xs_up_obw, xs_down_obw, xs_up_frac, xs_down_frac in zip(
            self.fake_edges[:-1],
            self.xs_over_bin_width,
            self.xs_up_over_bin_width,
            self.xs_down_over_bin_width,
            xs_up_fraction,
            xs_down_fraction,
        ):
            # Main plot
            ax.add_patch(
                Rectangle(
                    xy=(x+offset, xs_down_obw),
                    width=self.fake_bin_width / 2,
                    height=xs_up_obw - xs_down_obw,
                    hatch="/////",
                    fill=False,
                    color=color,
                    linewidth=0,
                )
            )

            # Ratio plot
            rax.add_patch(
                Rectangle(
                    xy=(x+offset, xs_down_frac),
                    width=self.fake_bin_width / 2,
                    height=xs_up_frac - xs_down_frac,
                    hatch="/////",
                    fill=False,
                    color=color,
                    linewidth=0,
                )
            )

        return ax, rax


class ObservableShapeFitted(ObservableShape):
    def plot(self, ax, rax, prediction_shape, bins_with_hline=None):
        if bins_with_hline is None:
            bins_with_hline = []

        color = category_specs[self.category]["color"]
        marker = category_specs[self.category]["marker"]

        ax.errorbar(
            self.fake_maybe_moved_centers,
            self.xs_over_bin_width,
            yerr=np.array(
                [
                    self.xs_over_bin_width - self.xs_down_over_bin_width,
                    self.xs_up_over_bin_width - self.xs_over_bin_width,
                ]
            ),
            linestyle="",
            color=color,
            marker=marker,
            markersize=5,
            capsize=3,
            label=category_specs[self.category]["plot_label"],
        )

        ratio_xs_over_bin_width = (
            self.xs_over_bin_width / prediction_shape.xs_over_bin_width
        )
        ratio_xs_up_over_bin_width = (
            self.xs_up_over_bin_width / prediction_shape.xs_up_over_bin_width
        )
        ratio_xs_down_over_bin_width = (
            self.xs_down_over_bin_width / prediction_shape.xs_down_over_bin_width
        )
        rax.errorbar(
            self.fake_maybe_moved_centers,
            ratio_xs_over_bin_width,
            yerr=np.array(
                [
                    ratio_xs_over_bin_width - ratio_xs_down_over_bin_width,
                    ratio_xs_up_over_bin_width - ratio_xs_over_bin_width,
                ]
            ),
            linestyle="",
            color=color,
            marker=marker,
            markersize=5,
            capsize=3,
        )

        for bin_index in bins_with_hline:
            ax.hlines(
                self.xs_over_bin_width[bin_index],
                self.fake_edges[bin_index],
                self.fake_edges[bin_index + 1],
                linestyle="dashed",
                linewidth=1,
                color=color,
            )
            rax.hlines(
                ratio_xs_over_bin_width[bin_index],
                self.fake_edges[bin_index],
                self.fake_edges[bin_index + 1],
                linestyle="dashed",
                linewidth=1,
                color=color,
            )

        return ax, rax

    def plot_as_band(self, ax, rax, prediction_shape):
        color = category_specs[self.category]["color"]
        label = category_specs[self.category]["plot_label"]
        fake_widths = np.diff(self.fake_edges)
        band_widths = fake_widths / 5

        # Main plot
        bottom_left_corners_x = self.fake_maybe_moved_centers - band_widths / 2
        heights = self.xs_up_over_bin_width - self.xs_down_over_bin_width

        lab = 0  # horrible, just horrible...
        for bottom_left_corner_x, bottom_left_corner_y, width, height in zip(
            bottom_left_corners_x, self.xs_down_over_bin_width, band_widths, heights
        ):
            ax.add_patch(
                Rectangle(
                    xy=(bottom_left_corner_x, bottom_left_corner_y),
                    width=width,
                    height=height,
                    color=color,
                    alpha=0.3,
                    label=f"{label}, Syst. unc." if lab == 0 else None,
                )
            )
            lab += 1

        # Ratio plot
        # This needs to be understood: in the case of up and down, should I divide by the nominal prediction
        # or by the nominal prediction + the up and down variations?
        ratio_xs_over_bin_width = (
            self.xs_over_bin_width / prediction_shape.xs_over_bin_width
        )
        ratio_xs_up_over_bin_width = (
            self.xs_up_over_bin_width / prediction_shape.xs_over_bin_width
        )
        ratio_xs_down_over_bin_width = (
            self.xs_down_over_bin_width / prediction_shape.xs_over_bin_width
        )

        if any(ratio_xs_down_over_bin_width > ratio_xs_over_bin_width):
            logger.warning("Ratio down is larger than ratio nominal!")
            logger.warning(f"Ratio down {ratio_xs_down_over_bin_width}")
            logger.warning(f"Ratio nominal {ratio_xs_over_bin_width}")
        if any(ratio_xs_up_over_bin_width < ratio_xs_over_bin_width):
            logger.warning("Ratio up is smaller than ratio nominal!")
            logger.warning(f"Ratio up {ratio_xs_up_over_bin_width}")
            logger.warning(f"Ratio nominal {ratio_xs_over_bin_width}")
        if any(ratio_xs_down_over_bin_width > ratio_xs_up_over_bin_width):
            logger.warning("Ratio down is larger than ratio up!")
            logger.warning(f"Ratio down {ratio_xs_down_over_bin_width}")
            logger.warning(f"Ratio up {ratio_xs_up_over_bin_width}")

        heights = ratio_xs_up_over_bin_width - ratio_xs_down_over_bin_width

        for bottom_left_corner_x, bottom_left_corner_y, width, height in zip(
            bottom_left_corners_x, ratio_xs_down_over_bin_width, band_widths, heights
        ):
            rax.add_patch(
                Rectangle(
                    xy=(bottom_left_corner_x, bottom_left_corner_y),
                    width=width,
                    height=height,
                    color=color,
                    alpha=0.3,
                )
            )

        return ax, rax

    def __sub__(self, other):
        if self.observable != other.observable or self.edges != other.edges:
            raise ValueError("Shapes are not compatible")

        # subtract relative uncertainties in quadrature
        rel_unc_up = np.abs(self.xs_up - self.xs) / self.xs
        rel_unc_down = np.abs(self.xs_down - self.xs) / self.xs
        rel_unc_up_other = np.abs(other.xs_up - other.xs) / other.xs
        rel_unc_down_other = np.abs(other.xs_down - other.xs) / other.xs
        
        new_xs_up = np.sqrt(rel_unc_up ** 2 - rel_unc_up_other ** 2) * self.xs
        new_xs_down = np.sqrt(rel_unc_down ** 2 - rel_unc_down_other ** 2) * self.xs

        return ObservableShapeFitted(
            self.observable,
            self.category,
            self.edges,
            self.xs,
            self.xs + new_xs_up,
            self.xs - new_xs_down, 
            self.overflow,
        )


class ObservableShapeKappa():
    """ A shape with no up and down variation, with ratio to SM already calculated
        and a redefinition of fake_rebin since the fake range can be squeezed instead of stretched
    """
    def __init__(
        self,
        parameters,
        edges,
        nominal_values,
        ratio_to_sm, 
    ):
        self.parameters = parameters
        self.edges = edges
        self.xs = nominal_values
        self.ratio_to_sm = ratio_to_sm
        self.fake_edges = np.arange(0, len(self.edges), 1)
        self.fake_centers = (self.fake_edges[1:] + self.fake_edges[:-1]) / 2

    def map_positions(self, arr, ref):
        mapped_positions = []
        for element in arr:
            closest_index = min(range(len(ref)), key=lambda i: abs(ref[i] - element))
            if closest_index == len(ref) - 1:
                mapped_positions.append(closest_index)
            else:
                mapped_positions.append(closest_index + (element - ref[closest_index]) / (ref[closest_index + 1] - ref[closest_index]))
        return mapped_positions

    def fake_rebin(self, other_shape):
        logger.debug("Now fake_rebinning for kappa")
        logger.debug(f"Edges: {self.edges}")
        logger.debug(f"Current fake edges: {self.fake_edges}")
        logger.debug(f"Other edges: {other_shape.edges}")
        logger.debug(f"Other fake edges: {other_shape.fake_edges}")
        self.fake_edges = []

        self.fake_edges = self.map_positions(self.edges, other_shape.edges)

        self.fake_edges = np.array(self.fake_edges)
        self.fake_centers = (self.fake_edges[1:] + self.fake_edges[:-1]) / 2
        logger.debug(f"New fake edges: {self.fake_edges}")

    def plot(self, ax, rax):
        color = "purple"

        lbl = ""
        for par, val in self.parameters.items():
            lbl += "{} = {}".format(TK_parameters_labels[par], val)
            # only add comma if it's not the last element
            if par != list(self.parameters.keys())[-1]:
                lbl += ", "
        ax.plot(self.fake_centers, self.xs, "o-", label=lbl, color=color)
        rax.plot(self.fake_centers, self.ratio_to_sm, "o-", color=color)
        return ax, rax


smH_PTH_Hgg_obs_shape = ObservableShapeSM(
    "smH_PTH",
    "Hgg",
    analyses_edges["smH_PTH"]["Hgg"],
    smH_PTH_Hgg_xs["central"].to_numpy(),
    smH_PTH_Hgg_xs["up"].to_numpy(),
    smH_PTH_Hgg_xs["down"].to_numpy(),
)

smH_PTH_MaximumGranularity_obs_shape = ObservableShapeSM(
    "smH_PTH",
    "Hgg",
    analyses_edges["smH_PTH"]["MaximumGranularity"],
    smH_PTH_MaximumGranularity_xs["central"].to_numpy(),
    smH_PTH_MaximumGranularity_xs["up"].to_numpy(),
    smH_PTH_MaximumGranularity_xs["down"].to_numpy(),
)

smH_PTH_EvenMoreMaximumGranularity_obs_shape = ObservableShapeSM(
    "smH_PTH",
    "Hgg",
    analyses_edges["smH_PTH"]["EvenMoreMaximumGranularity"],
    smH_PTH_EvenMoreMaximumGranularity_xs["central"].to_numpy(),
    smH_PTH_EvenMoreMaximumGranularity_xs["up"].to_numpy(),
    smH_PTH_EvenMoreMaximumGranularity_xs["down"].to_numpy(),
)

smH_PTH_HggHZZHWWHttHbb_obs_shape = deepcopy(smH_PTH_MaximumGranularity_obs_shape)
smH_PTH_HggHZZHWWHttHbb_obs_shape.category = "HggHZZHWWHttHbb"
smH_PTH_HggHZZHWWHttHbb_obs_shape.rebin(analyses_edges["smH_PTH"]["HggHZZHWWHttHbb"])

smH_PTH_HggHZZHWWHttHbbVBF_obs_shape = deepcopy(
    smH_PTH_EvenMoreMaximumGranularity_obs_shape
)
smH_PTH_HggHZZHWWHttHbbVBF_obs_shape.category = "HggHZZHWWHttHbbVBF"
smH_PTH_HggHZZHWWHttHbbVBF_obs_shape.rebin(
    analyses_edges["smH_PTH"]["HggHZZHWWHttHbbVBF"]
)

smH_PTH_FinalComb_obs_shape = deepcopy(smH_PTH_Hgg_obs_shape)
smH_PTH_FinalComb_obs_shape.category = "FinalComb"
smH_PTH_FinalComb_obs_shape.rebin(analyses_edges["smH_PTH"]["FinalComb"])


Njets_Hgg_obs_shape = ObservableShapeSM(
    "Njets",
    "Hgg",
    analyses_edges["Njets"]["Hgg"],
    Njets_Hgg_xs["central"].to_numpy(),
    Njets_Hgg_xs["up"].to_numpy(),
    Njets_Hgg_xs["down"].to_numpy(),
)

yH_Hgg_obs_shape = ObservableShapeSM(
    "yH",
    "Hgg",
    analyses_edges["yH"]["Hgg"],
    yH_Hgg_xs["central"].to_numpy(),
    yH_Hgg_xs["up"].to_numpy(),
    yH_Hgg_xs["down"].to_numpy(),
    overflow=False,
)

yH_Granular_obs_shape = ObservableShapeSM(
    "yH",
    "Hgg",
    analyses_edges["yH"]["Granular"],
    yH_Granular_xs["central"].to_numpy(),
    yH_Granular_xs["up"].to_numpy(),
    yH_Granular_xs["down"].to_numpy(),
    overflow=False,
)

smH_PTJ0_Hgg_obs_shape = ObservableShapeSM(
    "smH_PTJ0",
    "Hgg",
    analyses_edges["smH_PTJ0"]["Hgg"],
    smH_PTJ0_Hgg_xs["central"].to_numpy(),
    smH_PTJ0_Hgg_xs["up"].to_numpy(),
    smH_PTJ0_Hgg_xs["down"].to_numpy(),
)

smH_PTJ0_Granular_obs_shape = ObservableShapeSM(
    "smH_PTJ0",
    "Hgg",
    analyses_edges["smH_PTJ0"]["Granular"],
    smH_PTJ0_Granular_xs["central"].to_numpy(),
    smH_PTJ0_Granular_xs["up"].to_numpy(),
    smH_PTJ0_Granular_xs["down"].to_numpy(),
)

mjj_Hgg_obs_shape = ObservableShapeSM(
    "mjj",
    "Hgg",
    analyses_edges["mjj"]["Hgg"],
    mjj_Hgg_xs["central"].to_numpy(),
    mjj_Hgg_xs["up"].to_numpy(),
    mjj_Hgg_xs["down"].to_numpy(),
)

DEtajj_Hgg_obs_shape = ObservableShapeSM(
    "DEtajj",
    "Hgg",
    analyses_edges["DEtajj"]["Hgg"],
    DEtajj_Hgg_xs["central"].to_numpy(),
    DEtajj_Hgg_xs["up"].to_numpy(),
    DEtajj_Hgg_xs["down"].to_numpy(),
)

TauCJ_Hgg_obs_shape = ObservableShapeSM(
    "TauCJ",
    "Hgg",
    analyses_edges["TauCJ"]["Hgg"],
    TauCJ_Hgg_xs["central"].to_numpy(),
    TauCJ_Hgg_xs["up"].to_numpy(),
    TauCJ_Hgg_xs["down"].to_numpy(),
)

smH_PTH_Hgg_obs_shape_noNNLOPS = ObservableShapeSM(
    "smH_PTH",
    "Hgg",
    analyses_edges["smH_PTH"]["Hgg"],
    smH_PTH_Hgg_xs_noNNLOPS["central"].to_numpy(),
    smH_PTH_Hgg_xs_noNNLOPS["up"].to_numpy(),
    smH_PTH_Hgg_xs_noNNLOPS["down"].to_numpy(),
) 

smH_PTH_EvenMoreMaximumGranularity_obs_shape_noNNLOPS = ObservableShapeSM(
    "smH_PTH",
    "HggHZZHWWHttHbbVBF",
    analyses_edges["smH_PTH"]["EvenMoreMaximumGranularity"],
    smH_PTH_EvenMoreMaximumGranularity_xs_noNNLOPS["central"].to_numpy(),
    smH_PTH_EvenMoreMaximumGranularity_xs_noNNLOPS["up"].to_numpy(),
    smH_PTH_EvenMoreMaximumGranularity_xs_noNNLOPS["down"].to_numpy(),
)
smH_PTH_HggHZZHWWHttHbbVBF_obs_shape_noNNLOPS = deepcopy(
    smH_PTH_EvenMoreMaximumGranularity_obs_shape_noNNLOPS
)
smH_PTH_HggHZZHWWHttHbbVBF_obs_shape_noNNLOPS.rebin(
    analyses_edges["smH_PTH"]["HggHZZHWWHttHbbVBF"]
)

Njets_Hgg_obs_shape_noNNLOPS = ObservableShapeSM(
    "Njets",
    "Hgg",
    analyses_edges["Njets"]["Hgg"],
    Njets_Hgg_xs_noNNLOPS["central"].to_numpy(),
    Njets_Hgg_xs_noNNLOPS["up"].to_numpy(),
    Njets_Hgg_xs_noNNLOPS["down"].to_numpy(),
)

yH_Granular_obs_shape_noNNLOPS = ObservableShapeSM(
    "yH",
    "Hgg",
    analyses_edges["yH"]["Granular"],
    yH_Granular_xs_noNNLOPS["central"].to_numpy(),
    yH_Granular_xs_noNNLOPS["up"].to_numpy(),
    yH_Granular_xs_noNNLOPS["down"].to_numpy(),
    overflow=False,
)

smH_PTJ0_Hgg_obs_shape_noNNLOPS = ObservableShapeSM(
    "smH_PTJ0",
    "Hgg",
    analyses_edges["smH_PTJ0"]["Hgg"],
    smH_PTJ0_Hgg_xs_noNNLOPS["central"].to_numpy(),
    smH_PTJ0_Hgg_xs_noNNLOPS["up"].to_numpy(),
    smH_PTJ0_Hgg_xs_noNNLOPS["down"].to_numpy(),
) 

smH_PTJ0_Granular_obs_shape_noNNLOPS = ObservableShapeSM(
    "smH_PTJ0",
    "Hgg",
    analyses_edges["smH_PTJ0"]["Granular"],
    smH_PTJ0_Granular_xs_noNNLOPS["central"].to_numpy(),
    smH_PTJ0_Granular_xs_noNNLOPS["up"].to_numpy(),
    smH_PTJ0_Granular_xs_noNNLOPS["down"].to_numpy(),
)

mjj_Hgg_obs_shape_noNNLOPS = ObservableShapeSM(
    "mjj",
    "Hgg",
    analyses_edges["mjj"]["Hgg"],
    mjj_Hgg_xs_noNNLOPS["central"].to_numpy(),
    mjj_Hgg_xs_noNNLOPS["up"].to_numpy(),
    mjj_Hgg_xs_noNNLOPS["down"].to_numpy(),
)

DEtajj_Hgg_obs_shape_noNNLOPS = ObservableShapeSM(
    "DEtajj",
    "Hgg",
    analyses_edges["DEtajj"]["Hgg"],
    DEtajj_Hgg_xs_noNNLOPS["central"].to_numpy(),
    DEtajj_Hgg_xs_noNNLOPS["up"].to_numpy(),
    DEtajj_Hgg_xs_noNNLOPS["down"].to_numpy(),
)

TauCJ_Hgg_obs_shape_noNNLOPS = ObservableShapeSM(
    "TauCJ",
    "Hgg",
    analyses_edges["TauCJ"]["Hgg"],
    TauCJ_Hgg_xs_noNNLOPS["central"].to_numpy(),
    TauCJ_Hgg_xs_noNNLOPS["up"].to_numpy(),
    TauCJ_Hgg_xs_noNNLOPS["down"].to_numpy(),
)

smH_PTH_Hgg_obs_shape_powheg = ObservableShapeSM(
    "smH_PTH",
    "Hgg",
    analyses_edges["smH_PTH"]["Hgg"],
    smH_PTH_Hgg_xs_powheg["central"].to_numpy(),
    smH_PTH_Hgg_xs_powheg["up"].to_numpy(),
    smH_PTH_Hgg_xs_powheg["down"].to_numpy(),
)

smH_PTH_EvenMoreMaximumGranularity_obs_shape_powheg = ObservableShapeSM(
    "smH_PTH",
    "HggHZZHWWHttHbbVBF",
    analyses_edges["smH_PTH"]["EvenMoreMaximumGranularity"],
    smH_PTH_EvenMoreMaximumGranularity_xs_powheg["central"].to_numpy(),
    smH_PTH_EvenMoreMaximumGranularity_xs_powheg["up"].to_numpy(),
    smH_PTH_EvenMoreMaximumGranularity_xs_powheg["down"].to_numpy(),
)
smH_PTH_HggHZZHWWHttHbbVBF_obs_shape_powheg = deepcopy(
    smH_PTH_EvenMoreMaximumGranularity_obs_shape_powheg
)
smH_PTH_HggHZZHWWHttHbbVBF_obs_shape_powheg.rebin(
    analyses_edges["smH_PTH"]["HggHZZHWWHttHbbVBF"]
)

Njets_Hgg_obs_shape_powheg = ObservableShapeSM(
    "Njets",
    "Hgg",
    analyses_edges["Njets"]["Hgg"],
    Njets_Hgg_xs_powheg["central"].to_numpy(),
    Njets_Hgg_xs_powheg["up"].to_numpy(),
    Njets_Hgg_xs_powheg["down"].to_numpy(),
)

yH_Granular_obs_shape_powheg = ObservableShapeSM(
    "yH",
    "Hgg",
    analyses_edges["yH"]["Granular"],
    yH_Granular_xs_powheg["central"].to_numpy(),
    yH_Granular_xs_powheg["up"].to_numpy(),
    yH_Granular_xs_powheg["down"].to_numpy(),
    overflow=False,
)

smH_PTJ0_Hgg_obs_shape_powheg = ObservableShapeSM(
    "smH_PTJ0",
    "Hgg",
    analyses_edges["smH_PTJ0"]["Hgg"],
    smH_PTJ0_Hgg_xs_powheg["central"].to_numpy(),
    smH_PTJ0_Hgg_xs_powheg["up"].to_numpy(),
    smH_PTJ0_Hgg_xs_powheg["down"].to_numpy(),
)

smH_PTJ0_Granular_obs_shape_powheg = ObservableShapeSM(
    "smH_PTJ0",
    "Hgg",
    analyses_edges["smH_PTJ0"]["Granular"],
    smH_PTJ0_Granular_xs_powheg["central"].to_numpy(),
    smH_PTJ0_Granular_xs_powheg["up"].to_numpy(),
    smH_PTJ0_Granular_xs_powheg["down"].to_numpy(),
)

mjj_Hgg_obs_shape_powheg = ObservableShapeSM(
    "mjj",
    "Hgg",
    analyses_edges["mjj"]["Hgg"],
    mjj_Hgg_xs_powheg["central"].to_numpy(),
    mjj_Hgg_xs_powheg["up"].to_numpy(),
    mjj_Hgg_xs_powheg["down"].to_numpy(),
)

DEtajj_Hgg_obs_shape_powheg = ObservableShapeSM(
    "DEtajj",
    "Hgg",
    analyses_edges["DEtajj"]["Hgg"],
    DEtajj_Hgg_xs_powheg["central"].to_numpy(),
    DEtajj_Hgg_xs_powheg["up"].to_numpy(),
    DEtajj_Hgg_xs_powheg["down"].to_numpy(),
)

TauCJ_Hgg_obs_shape_powheg = ObservableShapeSM(
    "TauCJ",
    "Hgg",
    analyses_edges["TauCJ"]["Hgg"],
    TauCJ_Hgg_xs_powheg["central"].to_numpy(),
    TauCJ_Hgg_xs_powheg["up"].to_numpy(),
    TauCJ_Hgg_xs_powheg["down"].to_numpy(),
)

# It is assumed that the SM shapes are the ones with the finest binning, i.e. Hgg
# if something different will come up, we'll change it
sm_shapes = {
    "smH_PTH": smH_PTH_HggHZZHWWHttHbbVBF_obs_shape,
    #"smH_PTH": smH_PTH_Hgg_obs_shape,
    "Njets": Njets_Hgg_obs_shape,
    "yH": yH_Granular_obs_shape,
    "smH_PTJ0": smH_PTJ0_Granular_obs_shape,
    "mjj": mjj_Hgg_obs_shape,
    "DEtajj": DEtajj_Hgg_obs_shape,
    "TauCJ": TauCJ_Hgg_obs_shape,
}

sm_shapes_noNNLOPS = {
    "smH_PTH": smH_PTH_HggHZZHWWHttHbbVBF_obs_shape_noNNLOPS,
    #"smH_PTH": smH_PTH_Hgg_obs_shape_noNNLOPS,
    "Njets": Njets_Hgg_obs_shape_noNNLOPS,
    "yH": yH_Granular_obs_shape_noNNLOPS,
    "smH_PTJ0": smH_PTJ0_Granular_obs_shape_noNNLOPS,
    "mjj": mjj_Hgg_obs_shape_noNNLOPS,
    "DEtajj": DEtajj_Hgg_obs_shape_noNNLOPS,
    "TauCJ": TauCJ_Hgg_obs_shape_noNNLOPS,
}

sm_shapes_powheg = {
    "smH_PTH": smH_PTH_HggHZZHWWHttHbbVBF_obs_shape_powheg,
    #"smH_PTH": smH_PTH_Hgg_obs_shape_powheg,
    "Njets": Njets_Hgg_obs_shape_powheg,
    "yH": yH_Granular_obs_shape_powheg,
    "smH_PTJ0": smH_PTJ0_Granular_obs_shape_powheg,
    "mjj": mjj_Hgg_obs_shape_powheg,
    "DEtajj": DEtajj_Hgg_obs_shape_powheg,
    "TauCJ": TauCJ_Hgg_obs_shape_powheg,
}