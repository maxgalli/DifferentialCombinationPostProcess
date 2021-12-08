import uproot
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from warnings import warn

import logging
logger = logging.getLogger(__name__)



class DifferentialSpectrum:
    """ Basically a collection of Scan instances, one per POI, for a single category 
    """
    def __init__(self, variable, category, pois, input_dirs):
        logger.info("Building a DifferentialSpectrum for variable {} and category {} with the following POIs {}".format(variable, category, pois))
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
                logger.warning("Attempted to fetch scan results for POI {}, but no files where found".format(poi))
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
        logger.info("Looking for files that look like {} inside {}".format(
            self.file_name_tmpl, input_dirs)
            )
        # If a file is corrupted, uproot will raise an error; it will have to be manually removed
        branches = uproot.concatenate(
            ["{}/{}:{}".format(input_dir, self.file_name_tmpl, self.tree_name) for input_dir in input_dirs],
            expressions=[self.poi, "deltaNLL"],
            cut=self.default_cut
        )
        logger.info("Found {} points".format(len(branches)))

        # For semplicity, make two separate arrays after sorting them
        branches_np = np.array([branches[poi], branches["deltaNLL"]])
        branches_np = branches_np[:, branches_np[0].argsort()]
        poi_values_original = branches_np[0]
        two_dnll_original = 2 * branches_np[1]
        # Points will be arranged in a 2xN_Points numpy array
        self.original_points = np.array([poi_values_original, two_dnll_original])
        logger.debug("Original points found: {}".format([
            (x, y) for x, y in zip(self.original_points[0], self.original_points[1])]))

        # Interpolate and re-make arrays to have more points
        self.dNLL_func = interpolate.interp1d(poi_values_original, two_dnll_original)
        self.n_interpol = 1000000
        self.poi_boundaries = (poi_values_original[0], poi_values_original[-1])
        poi_values = np.linspace(self.poi_boundaries[0], self.poi_boundaries[1], self.n_interpol)
        two_dnll = self.dNLL_func(poi_values)
        self.interpolated_points = np.array([poi_values, two_dnll])

        # Check if there are nan values in two_dnnl
        if np.isnan(self.interpolated_points[1]).any():
            logger.warning("NaN values detected in NLLs values, removing them")
            self.interpolated_points = self.interpolated_points[:, ~np.isnan(self.interpolated_points[1])]

        # Find minimum and compute uncertainties
        self.find_minimum()
        logger.info("Found minimum at ({}, {})".format(self.minimum[0], self.minimum[1]))
        self.compute_uncertainties()
        logger.info("Down uncertainty: {}, up uncertainty: {}".format(
            self.down_uncertainty, self.up_uncertainty
            )
        )

    
    def find_minimum(self):
        # Minimum will be a numpy array of shape (2,)
        self.minimum = self.interpolated_points[:, np.argmin(self.interpolated_points[1])]


    def compute_uncertainties(self):
        level = self.minimum[1] + 1.
        level_arr = np.ones(len(self.interpolated_points[1])) * level
        # Get index of the two points in poi_values where the NLL crosses the horizontal line at 1
        try:
            down_idx, up_idx = np.argwhere(np.diff(np.sign(self.interpolated_points[1] - level_arr))).flatten()
            self.down = self.interpolated_points[:, down_idx]
            self.up = self.interpolated_points[:, up_idx]
            self.down_uncertainty = abs(self.minimum[0] - self.down[0])
            self.up_uncertainty = abs(self.minimum[0] - self.up[0])

        except ValueError as e:
            # If this is the case, set up and down to the minimum and the uncertainties to 0, so it gets plotted anyways
            warn(
                "The NLL curve does not seem to cross the horizontal line. Try scanning a wider range of points for {}!\n\
                Original error: {}".format(self.poi, e)
                )
            logger.info("Setting up and down to minimum and uncertainties to 0.")
            self.down = self.minimum
            self.up = self.minimum
            self.down_uncertainty = 0.
            self.up_uncertainty = 0.


    def plot(self, ax, color=None):
        logger.info("Plotting scan for {}".format(self.poi))
        # Restrict the plotted values to a dnll less than 3.5
        x = self.interpolated_points[0][self.interpolated_points[1] < 3.5]
        y = self.interpolated_points[1][self.interpolated_points[1] < 3.5]
        ax.plot(x, y, color="k")
        # Vertical line passing through the minimum
        ax.plot(
            [self.minimum[0], self.minimum[0]], [self.minimum[1], self.up[1]], 
            color=color,
            linestyle="--",
            label=self.poi
        )
        # Points where NLL crosses 1
        ax.plot(
            [self.down[0], self.up[0]], [self.down[1], self.up[1]], 
            color=color,
            linestyle="",
            marker="o"
            )
        
        return ax