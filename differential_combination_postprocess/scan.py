from cProfile import label
import uproot
import glob
import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from warnings import warn
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)


class DifferentialSpectrum:
    """ Basically a collection of Scan instances, one per POI, for a single category 
    """

    def __init__(self, variable, category, pois, input_dirs):
        logger.info(
            "Building a DifferentialSpectrum for variable {} and category {} with the following POIs {}".format(
                variable, category, pois
            )
        )
        self.variable = variable
        self.category = category
        self.scans = {}
        for poi in pois:
            try:
                self.scans[poi] = Scan(poi, input_dirs)
            # this is the case in which there are no scans for poi in input_dir, but we are looking
            # for them anyways because the list of pois is taken from the metadata
            # IOError is raised by uproot.concatenate when no files matching the regex are found
            except IOError:
                logger.warning(
                    "Attempted to fetch scan results for POI {}, but no files where found".format(
                        poi
                    )
                )
                pass


class Scan:
    """
    """

    def __init__(self, poi, input_dirs):
        self.default_cut = "deltaNLL<990.0"
        self.file_name_tmpl = "higgsCombine_SCAN_{}*.root".format(poi)
        self.tree_name = "limit"
        self.poi = poi

        # Read the two branches we are interested in: deltaNLL and the POI one
        logger.info(
            "Looking for files that look like {} inside {}".format(
                self.file_name_tmpl, input_dirs
            )
        )
        # If a file is corrupted, uproot will raise an error; it will have to be manually removed
        branches = uproot.concatenate(
            [
                "{}/{}:{}".format(input_dir, self.file_name_tmpl, self.tree_name)
                for input_dir in input_dirs
            ],
            expressions=[self.poi, "deltaNLL"],
            # cut=self.default_cut,
        )
        logger.info("Found {} points".format(len(branches)))

        """
        # IMPORTANT: on 21 Feb, while tryin Hgg pt, it looks like many files have two 'events':
        # one in which deltaNLL is 0, which fucks up the deltaNLL profile, and a good one
        # we thus decide to remove the rows for which deltaNLL is 0 ONLY IF MORE THAN ONE ROW HAS deltaNLL=0
        # this is like this otherwise HZZ (in which the minimum is actually 0) would be ruined
        logger.debug(
            "Original points found: {}".format(
                [(x, y) for x, y in zip(branches[self.poi], branches.deltaNLL)]
            )
        )
        dnll_zeros = branches.deltaNLL[branches.deltaNLL == 0]
        if len(dnll_zeros) > 1:
            logger.info("Removing {} rows with deltaNLL=0".format(len(dnll_zeros)))
            branches = branches[branches.deltaNLL != 0]
        """

        # For simplicity, make two separate arrays after sorting them
        branches_np = np.array([branches[poi], branches["deltaNLL"]])
        branches_np = branches_np[:, branches_np[0].argsort()]

        # Some kinds of interpolation (like cubic) fail if there are duplicates (these may come e.g. from multiple directories containing the scans)
        # We thus remove them
        branches_np = np.unique(branches_np, axis=-1)

        # At this point we can still have duplicates, most likely if there are same values for the POI with different deltaNLL
        for val in np.unique(branches_np[0]):
            wh = np.where(branches_np[0] == val)[0]
            if len(wh) > 1:
                branches_np = np.delete(branches_np, wh[1:], axis=1)

        poi_values_original = branches_np[0]
        two_dnll_original = 2 * branches_np[1]

        # Points will be arranged in a 2xN_Points numpy array
        self.original_points = np.array([poi_values_original, two_dnll_original])

        logger.debug(
            "Original points found (after sorting, removing zeroes and removing duplicates): {}".format(
                [
                    (x, y)
                    for x, y in zip(
                        self.original_points[0], self.original_points[1] / 2
                    )
                ]
            )
        )

        # Interpolate and re-make arrays to have more points
        """ For my future self and in case of problems during the presentation of results: 
        interp1d with cubic should perform a spline third order interpolation, exactly like the ones 
        performed with ROOT
        scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
        ROOT: https://root.cern.ch/doc/master/classTSpline3.html#ac94f978dc582faf55dcb574003b8fdeb
        """
        self.dNLL_func = interpolate.interp1d(
            poi_values_original, two_dnll_original, kind="cubic"
        )
        self.n_interpol = 1000000
        self.poi_boundaries = (poi_values_original[0], poi_values_original[-1])
        poi_values = np.linspace(
            self.poi_boundaries[0], self.poi_boundaries[1], self.n_interpol
        )
        two_dnll = self.dNLL_func(poi_values)
        self.interpolated_points = np.array([poi_values, two_dnll])

        # Check if there are nan values in two_dnnl
        if np.isnan(self.interpolated_points[1]).any():
            logger.warning("NaN values detected in NLLs values, removing them")
            self.interpolated_points = self.interpolated_points[
                :, ~np.isnan(self.interpolated_points[1])
            ]

        # Find minimum and compute uncertainties
        self.find_minimum()
        self.original_moved_points = deepcopy(self.original_points)
        if (
            abs(self.minimum[1]) > 0.0001
        ):  # Manually move the minimum to 0 if it is not already there
            logger.info(
                "dnll of minimum was found to be at more than 0.0001 (i.e. not at 0). Will be manually set to 0"
            )
            offset = self.minimum[1]
            self.original_moved_points[1] -= offset
            self.interpolated_points[1] -= offset
            self.find_minimum()
        logger.info(
            "Found minimum at ({}, {})".format(self.minimum[0], self.minimum[1])
        )
        self.down68, self.up68, self.down68_unc, self.up68_unc = self.compute_uncertainties(
            1.0
        )
        self.down95, self.up95, self.down95_unc, self.up95_unc = self.compute_uncertainties(
            4.0
        )
        logger.info(
            "Down uncertainty 1sigma: {}, up uncertainty 1sigma: {}".format(
                self.down68_unc, self.up68_unc
            )
        )

    def find_minimum(self):
        # Minimum will be a numpy array of shape (2,)
        self.minimum = self.interpolated_points[
            :, np.argmin(self.interpolated_points[1])
        ]

    def compute_uncertainties(self, level=1.0):
        level = self.minimum[1] + level
        level_arr = np.ones(len(self.interpolated_points[1])) * level
        # Get index of the two points in poi_values where the NLL crosses the horizontal line at 1
        indices = np.argwhere(
            np.diff(np.sign(self.interpolated_points[1] - level_arr))
        ).flatten()
        points = [self.interpolated_points[:, i] for i in indices]
        logger.debug(f"Points where NLL crosses the horizontal line at 1: {points}")
        if len(points) < 2:
            # If this is the case, set up and down to the minimum and the uncertainties to 0, so it gets plotted anyways
            logger.warning(
                f"The NLL curve does not seem to cross the horizontal line. Try scanning a wider range of points for {self.poi}!"
            )
            logger.info("Setting up and down to minimum and uncertainties to 0.")
            down = self.minimum
            up = self.minimum
            down_uncertainty = 0.0
            up_uncertainty = 0.0
        elif len(points) > 2:
            logger.warning(
                "More than two points where NLL crosses the horizontal line at 1. First and last will be used."
            )
            down_idx = indices[0]
            up_idx = indices[-1]
            down = self.interpolated_points[:, down_idx]
            up = self.interpolated_points[:, up_idx]
            down_uncertainty = abs(self.minimum[0] - down[0])
            up_uncertainty = abs(self.minimum[0] - up[0])
        else:
            down_idx, up_idx = indices
            down = self.interpolated_points[:, down_idx]
            up = self.interpolated_points[:, up_idx]
            down_uncertainty = abs(self.minimum[0] - down[0])
            up_uncertainty = abs(self.minimum[0] - up[0])

        return down, up, down_uncertainty, up_uncertainty

    def plot(self, ax, color=None, label=None, ylim=8.0):
        if label is None:
            label = self.poi
        # Restrict the plotted values to a dnll less than ylim
        x = self.interpolated_points[0][self.interpolated_points[1] < ylim]
        y = self.interpolated_points[1][self.interpolated_points[1] < ylim]
        ax.plot(x, y, color="k")
        # Vertical line passing through the minimum
        ax.plot(
            [self.minimum[0], self.minimum[0]],
            [self.minimum[1], self.up68[1]],
            color=color,
            linestyle="--",
            label=label,
        )
        # Vertical line passing through down68
        ax.plot(
            [self.down68[0], self.down68[0]],
            [self.minimum[1], self.down68[1]],
            color="k",
            linestyle="--",
        )
        # Vertical line passing through up68
        ax.plot(
            [self.up68[0], self.up68[0]],
            [self.minimum[1], self.up68[1]],
            color="k",
            linestyle="--",
        )
        # Vertical line passing through down95
        ax.plot(
            [self.down95[0], self.down95[0]],
            [self.minimum[1], self.down95[1]],
            color="k",
            linestyle="--",
        )
        # Vertical line passing through up95
        ax.plot(
            [self.up95[0], self.up95[0]],
            [self.minimum[1], self.up95[1]],
            color="k",
            linestyle="--",
        )
        # Points where NLL crosses 1
        ax.plot(
            [self.down68[0], self.up68[0]],
            [self.down68[1], self.up68[1]],
            color=color,
            linestyle="",
            marker="o",
        )
        # Points where NLL crosses 4
        ax.plot(
            [self.down95[0], self.up95[0]],
            [self.down95[1], self.up95[1]],
            color=color,
            linestyle="",
            marker="o",
        )

        return ax

    def plot_original_points(self, ax, color=None, label=None, for_single_plot=False):
        if label is None:
            label = f"{self.poi} (original)"
        if for_single_plot:
            x = self.original_moved_points[0]
            y = self.original_moved_points[1]
        else:
            x = self.original_moved_points[0][self.original_moved_points[1] < 8.0]
            y = self.original_moved_points[1][self.original_moved_points[1] < 8.0]
        ax.plot(x, y, color=color, linestyle="", marker="*", label=label)

        return ax


class Scan2D:
    def __init__(self, pois, file_name_template, input_dirs):
        if pois is None:
            raise ValueError("pois must be a list of strings")
        self.pois = pois
        self.file_name_template = file_name_template
        self.default_cut = "deltaNLL<990.0"
        self.tree_name = "limit"

        branches = uproot.concatenate(
            [
                "{}/{}:{}".format(input_dir, self.file_name_template, self.tree_name)
                for input_dir in input_dirs
            ],
            expressions=[*pois, "deltaNLL"],
            cut=self.default_cut,
            library="np",
        )

        x = branches[pois[0]]
        y = branches[pois[1]]
        z = 2 * branches["deltaNLL"]

        self.points = np.array([x, y, z])
        self.minimum = self.points[:, np.argmin(z)]

        self.y_int, self.x_int = np.mgrid[
            y.min() : y.max() : 1000j, x.min() : x.max() : 1000j
        ]
        self.z_int = griddata((x, y), z, (self.x_int, self.y_int), method="cubic")

        self.z_int[0] -= self.z_int.min()
        self.z_int[1] -= self.z_int.min()

    def plot_as_heatmap(self, ax):
        colormap = plt.get_cmap("Oranges")
        colormap = colormap.reversed()
        pc = ax.pcolormesh(
            self.x_int, self.y_int, self.z_int, cmap=colormap, shading="gouraud"
        )

        return ax, colormap, pc

    def plot_as_contour(self, ax, color="k"):
        cs = ax.contour(
            self.x_int,
            self.y_int,
            self.z_int,
            levels=[1.0],
            colors=[color],
            linewidths=[2.0],
        )
        # Best value as point
        ax.plot(
            [self.minimum[0]], [self.minimum[1]], color=color, linestyle="", marker="o"
        )

        return ax
