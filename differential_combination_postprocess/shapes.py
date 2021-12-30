import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import cycle
from copy import deepcopy

from .physics import YR4_totalXS
from .physics import smH_PTH_Hgg_xs, analyses_edges
from .cosmetics import markers, category_specs
from .utils import merge_bins

import logging
logger = logging.getLogger(__name__)


class ObservableShape:
    """
    """
    def __init__(
        self, observable, edges, 
        nominal_values, up_values, down_values, 
        overflow=True
        ):
        
        self.observable = observable
        self.edges = edges
        self.xs = nominal_values
        self.xs_up = up_values
        self.xs_down = down_values
        self.overflow = overflow
        
        # Since we want every bin of the same width, we can do this for the steps
        self.fake_bin_width = 1
        self.fake_edges = np.arange(0, len(self.edges), self.fake_bin_width)
        self.fake_centers = (self.fake_edges[1:] + self.fake_edges[:-1]) / 2


    @property
    def nbins(self):
        return len(self.xs)


    @property
    def inclusive_xs(self):
        return np.sum(self.xs)


    def rebin(self, new_edges):
        self.xs = merge_bins(self.xs, self.edges, new_edges)
        self.xs_up = merge_bins(self.xs_up, self.edges, new_edges)
        self.xs_down = merge_bins(self.xs_down, self.edges, new_edges)
        self.edges = new_edges


    def fake_rebin(self, other_shape):
        logger.debug(f"Stretching fake range for {self.observable}")
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


    def compute_xs_over_bin_width(self, xs):
        if self.overflow:
            xs_o_bw = [xs[i] / (self.edges[i+1] - self.edges[i]) for i in range(self.nbins - 1)] + [xs[-1]]
            # Normalize by second to last bin
            xs_o_bw[-1] = xs[-1] / (self.edges[-2] - self.edges[-3])
        else:
            xs_o_bw = [xs[i] / (self.edges[i+1] - self.edges[i]) for i in range(self.nbins - 1)]
        
        return np.array(xs_o_bw)


    def compute_unc_xs_over_bin_width(self, unc_fraction):
        return [xs * unc for xs, unc in zip(self.xs, unc_fraction)]


    def __str__(self):
        string = f"Observable: {self.observable}\n"
        string += f"Edges: {self.edges}\n"
        string += f"Nominal XS: {self.xs}\n"
        string += f"Up XS: {self.xs_up}\n"
        string += f"Down XS: {self.xs_down}\n"
        
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
            values=self.xs,
            color="grey",
            linewidth=1,
            label="aMC@NLO"
            )
        
        rax.axhline(y=1, color="grey", linewidth=1)
        
        # Apply rectangular patches for uncertainties
        xs_up_fraction = self.xs_up / self.xs
        xs_down_fraction = self.xs_down / self.xs
        for x, xs, xs_up, xs_down, xs_up_frac, xs_down_frac in zip(
            self.fake_edges[:-1], 
            self.xs, 
            self.xs_up,
            self.xs_down,
            xs_up_fraction,
            xs_down_fraction
            ):
            # Main plot
            ax.add_patch(Rectangle(
                xy=(x, xs_down), 
                width=self.fake_bin_width,
                height=xs_up - xs_down,
                hatch="/////",
                fill=False,
                color="grey",
                linewidth=0)
                )
            # Ratio plot
            rax.add_patch(Rectangle(
                xy=(x, xs_down_frac),
                width=1,
                height=xs_up_frac - xs_down_frac,
                hatch="/////",
                fill=False,
                color="grey",
                linewidth=0)
                )

        return ax, rax


class ObservableShapeFitted(ObservableShape):
    def plot(self, ax, rax, prediction_shape, color="black", dashed_horizontal_lines=False):
        markers_iter = cycle(markers)
        marker = next(markers_iter)
        
        ax.errorbar(
            self.fake_centers,
            self.xs,
            yerr = np.array(
                [self.xs - self.xs_down, 
                self.xs_up - self.xs]
                ),
            linestyle="",
            color=color,
            marker=marker,
            markersize=5,
            capsize=3,
        )

        ratio_xs = self.xs / prediction_shape.xs
        ratio_xs_up = self.xs_up / prediction_shape.xs
        ratio_xs_down = self.xs_down / prediction_shape.xs
        rax.errorbar(
            self.fake_centers,
            ratio_xs,
            yerr = np.array(
                [ratio_xs - ratio_xs_down,
                ratio_xs_up - ratio_xs]
                ),
            linestyle="",
            color=color,
            marker=marker,
            markersize=5,
            capsize=3,
        )

        if dashed_horizontal_lines:
            for y, left, right in zip(self.xs, self.fake_edges[:-1], self.fake_edges[1:]):
                ax.hlines(y, left, right, linestyle="dashed", linewidth=1, color=color)
            for y, left, right in zip(ratio_xs, self.fake_edges[:-1], self.fake_edges[1:]):
                rax.hlines(y, left, right, linestyle="dashed", linewidth=1, color=color)

        return ax, rax


smH_PTH_Hgg_obs_shape = ObservableShapeSM(
    "smH_PTH",
    analyses_edges["smH_PTH"]["Hgg"], 
    smH_PTH_Hgg_xs["central"].to_numpy(), 
    smH_PTH_Hgg_xs["up"].to_numpy(), 
    smH_PTH_Hgg_xs["down"].to_numpy()
    )

# It is assumed that the SM shapes are the ones with the finest binning, i.e. Hgg
# if something different will come up, we'll change it
sm_shapes = {
    "smH_PTH": smH_PTH_Hgg_obs_shape,
}