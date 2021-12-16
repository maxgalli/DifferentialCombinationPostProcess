import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .physics import YR4_totalXS
from .physics import smH_PTH_Hgg_xs, analyses_edges
from .cosmetics import markers

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
        self.inclusive_xs = np.sum(self.xs)
        self.nbins = len(self.xs)
        self.overflow = overflow
        
        self.xs_over_bin_width = self.xs
        self.xs_over_bin_width_up = self.xs_up
        self.xs_over_bin_width_down = self.xs_down
        
        self.xs_over_bin_width_up_fraction = self.xs_over_bin_width_up / self.xs_over_bin_width
        self.xs_over_bin_width_down_fraction = self.xs_over_bin_width_down / self.xs_over_bin_width

        # Since we want every bin of the same width, we can do this for the steps
        self.fake_bin_width = 1
        self.fake_edges = np.arange(0, len(self.edges), self.fake_bin_width)
        self.fake_centers = (self.fake_edges[1:] + self.fake_edges[:-1]) / 2


    def stretch_fake_range(self, other_shape):
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
        return [xs * unc for xs, unc in zip(self.xs_over_bin_width, unc_fraction)]



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
            linewidth=2
            )
        
        rax.axhline(y=1, color="grey", linewidth=2)
        
        # Apply rectangular patches for uncertainties
        for x, xs, xs_up, xs_down, xs_up_frac, xs_down_frac in zip(
            self.fake_edges[:-1], 
            self.xs_over_bin_width, 
            self.xs_over_bin_width_up,
            self.xs_over_bin_width_down,
            self.xs_over_bin_width_up_fraction,
            self.xs_over_bin_width_down_fraction
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
    def plot(self, ax, rax, color="black"):
        marker = next(markers)
        ax.errorbar(
            self.fake_centers,
            self.xs_over_bin_width,
            yerr = np.array(
                [self.xs_over_bin_width - self.xs_over_bin_width_down, 
                self.xs_over_bin_width_up - self.xs_over_bin_width]
                ),
            linestyle="",
            color=color,
            marker=marker,
            markersize=10,
        )

        return ax, rax


logger.debug(f"smH_PTH_Hgg_xs first bin: {smH_PTH_Hgg_xs[0]}")
smH_PTH_Hgg_obs_shape = ObservableShapeSM(
    "smH_PTH",
    analyses_edges["smH_PTH"]["Hgg"], 
    smH_PTH_Hgg_xs["central"].to_numpy(), 
    smH_PTH_Hgg_xs["up"].to_numpy(), 
    smH_PTH_Hgg_xs["down"].to_numpy()
    )

sm_shapes = {
    "smH_PTH": smH_PTH_Hgg_obs_shape,
}