from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import cycle
from copy import deepcopy

from .physics import YR4_totalXS
from .physics import smH_PTH_Hgg_xs, Njets_Hgg_xs, yH_Hgg_xs, analyses_edges
from .cosmetics import markers, category_specs

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

    def fake_rebin(self, other_shape):
        logger.debug(f"Stretching fake range for {self.observable} {self.category}")
        logger.debug(f"Current Fake edges: {self.fake_edges}")
        logger.debug(f"Other Fake edges: {other_shape.fake_edges}")
        self.fake_edges = []
        self.fake_centers = []
        new_unit = other_shape.fake_bin_width
        new_edge = other_shape.fake_edges[0]
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

    def merge_bins(self, old_bins, old_edges, new_edges):
        if not all(edge in old_edges for edge in new_edges):
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
            label="aMC@NLO",
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


class ObservableShapeFitted(ObservableShape):
    def plot(self, ax, rax, prediction_shape, dashed_horizontal_lines=False):
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

        if dashed_horizontal_lines:
            for y, left, right in zip(
                self.xs_over_bin_width, self.fake_edges[:-1], self.fake_edges[1:]
            ):
                ax.hlines(y, left, right, linestyle="dashed", linewidth=1, color=color)
            for y, left, right in zip(
                ratio_xs_over_bin_width, self.fake_edges[:-1], self.fake_edges[1:]
            ):
                rax.hlines(y, left, right, linestyle="dashed", linewidth=1, color=color)

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
        ratio_xs_over_bin_width = (
            self.xs_over_bin_width / prediction_shape.xs_over_bin_width
        )
        ratio_xs_up_over_bin_width = (
            self.xs_up_over_bin_width / prediction_shape.xs_up_over_bin_width
        )
        ratio_xs_down_over_bin_width = (
            self.xs_down_over_bin_width / prediction_shape.xs_down_over_bin_width
        )

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

        return ObservableShapeFitted(
            self.observable,
            self.category,
            self.edges,
            self.xs,
            self.xs + np.abs(self.xs_up - other.xs_up),
            self.xs - np.abs(self.xs_down - other.xs_down),
            self.overflow,
        )


smH_PTH_Hgg_obs_shape = ObservableShapeSM(
    "smH_PTH",
    "Hgg",
    analyses_edges["smH_PTH"]["Hgg"],
    smH_PTH_Hgg_xs["central"].to_numpy(),
    smH_PTH_Hgg_xs["up"].to_numpy(),
    smH_PTH_Hgg_xs["down"].to_numpy(),
)

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


# It is assumed that the SM shapes are the ones with the finest binning, i.e. Hgg
# if something different will come up, we'll change it
sm_shapes = {
    "smH_PTH": smH_PTH_Hgg_obs_shape,
    "Njets": Njets_Hgg_obs_shape,
    "yH": yH_Hgg_obs_shape,
}

