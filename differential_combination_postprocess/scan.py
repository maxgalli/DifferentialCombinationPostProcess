from cProfile import label
import uproot
import awkward as ak
import glob
import os
import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from warnings import warn
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter

import logging

logger = logging.getLogger(__name__)

from differential_combination_postprocess.utils import (
    truncate_colormap,
    custom_colormap,
)
from differential_combination_postprocess.cosmetics import (
    get_parameter_label
)

uproot_main_version = int(uproot.__version__.split(".")[0])


class DifferentialSpectrum:
    """Basically a collection of Scan instances, one per POI, for a single category"""

    def __init__(
        self,
        variable,
        category,
        pois,
        input_dirs,
        from_singles=False,
        skip_best=False,
        file_name_tmpl=None,
        cut_strings=None,
        allow_extrapolation=True,
    ):
        logger.info(
            "Building a DifferentialSpectrum for variable {} and category {} with the following POIs {}".format(
                variable, category, pois
            )
        )
        logger.debug(f"cut_strings: {cut_strings}")
        self.variable = variable
        self.category = category
        self.scans = {}
        self.from_singles = from_singles
        if cut_strings is None:
            cut_strings = {}
        for poi in pois:
            try:
                which_scan = ScanSingles if from_singles else Scan
                self.scans[poi] = which_scan(
                    poi,
                    input_dirs,
                    skip_best=skip_best,
                    file_name_tmpl=file_name_tmpl,
                    cut_strings=cut_strings[poi] if poi in cut_strings else None,
                    allow_extrapolation=allow_extrapolation,
                )
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
    """ """

    def __init__(
        self,
        poi,
        input_dirs,
        skip_best=False,
        file_name_tmpl=None,
        extra_selections=None,
        cut_strings=None,
        allow_extrapolation=True,
    ):
        if cut_strings is None:
            cut_strings = []
        logger.debug(f"cut_strings: {cut_strings}")
        self.default_cut = "deltaNLL<990.0"
        if file_name_tmpl is None:
            self.file_name_tmpl = "higgsCombine_SCAN_{}*.root".format(poi)
        else:
            self.file_name_tmpl = file_name_tmpl
        self.tree_name = "limit"
        self.poi = poi
        self.allow_extrapolation = allow_extrapolation

        # Read the two branches we are interested in: deltaNLL and the POI one
        logger.info(
            "Looking for files that look like {} inside {}".format(
                self.file_name_tmpl, input_dirs
            )
        )
        # If a file is corrupted, uproot will raise an error; it will have to be manually removed
        dirs_template = [
            "{}/{}:{}".format(input_dir, self.file_name_tmpl, self.tree_name)
            for input_dir in input_dirs
        ]
        if skip_best:
            to_concatenate = []
            # from uproot 5 iterate returns each event as a dict, while before it was a tuple
            if uproot_main_version > 4:
                # glob names of all files
                files = []
                for dr in dirs_template:
                    dr = dr.split(":")[0]
                    files.extend(glob.glob(dr))
                for f in files:
                    # open file
                    f = uproot.open(f)
                    # get tree
                    tree = f[self.tree_name]
                    # get branches
                    batch = tree.arrays([self.poi, "deltaNLL"])
                    to_concatenate.append(batch[1:])
            else:
                for batch in uproot.iterate(dirs_template):
                    batch = batch[[self.poi, "deltaNLL"]]
                    to_concatenate.append(batch[1:])
            branches = ak.concatenate(to_concatenate)
        else:
            branches = uproot.concatenate(
                dirs_template,
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

        # Apply extra selections if present
        mask = np.ones(len(poi_values_original), dtype=bool)
        if extra_selections is not None:
            logger.info(
                "Applying extra selections for POI {}: {}".format(
                    poi, extra_selections(poi_values_original)
                )
            )
            mask = extra_selections(poi_values_original)
        poi_values_original = poi_values_original[mask]
        two_dnll_original = two_dnll_original[mask]

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

        for cut_string in cut_strings:
            self.cut_from_string(cut_string)

        logger.debug(
            "Original points found (after applying custom cuts): {}".format(
                [
                    (x, y)
                    for x, y in zip(
                        self.original_points[0], self.original_points[1] / 2
                    )
                ]
            )
        )

        # Check if there are nan values in original points
        if np.isnan(self.original_points[1]).any():
            logger.warning("NaN values detected in NLLs values, removing them")
            self.original_points = self.original_points[
                :, ~np.isnan(self.original_points[1])
            ]

        # Interpolate and re-make arrays to have more points
        """ For my future self and in case of problems during the presentation of results: 
        interp1d with cubic should perform a spline third order interpolation, exactly like the ones 
        performed with ROOT
        scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
        ROOT: https://root.cern.ch/doc/master/classTSpline3.html#ac94f978dc582faf55dcb574003b8fdeb
        """
        self.dNLL_func = interpolate.interp1d(
            # poi_values_original, two_dnll_original, kind="cubic"
            self.original_points[0],
            self.original_points[1],
            kind="cubic",
        )
        self.n_interpol = 1000000
        # self.poi_boundaries = (poi_values_original[0], poi_values_original[-1])
        self.poi_boundaries = (self.original_points[0][0], self.original_points[0][-1])
        poi_values = np.linspace(
            self.poi_boundaries[0], self.poi_boundaries[1], self.n_interpol
        )
        two_dnll = self.dNLL_func(poi_values)
        self.interpolated_points = np.array([poi_values, two_dnll])

        self.check_points_and_compute_all_uncertainties()

        # If up68 or down68 are 0, it means that NLL does not cross 1.0 at all
        # In this case we repeat the interpolation procedure in a wider range with UnivariateSpline
        #if (self.up68_unc[0] == 0 or self.down68_unc[0] == 0) and self.allow_extrapolation:
        if (self.up95_unc[0] == 0 or self.down95_unc[0] == 0) and self.allow_extrapolation:
            logger.warning(
                "NLL does not cross 1.0 at all. Will try to interpolate with a wider range"
            )
            self.dNLL_func = UnivariateSpline(
                self.original_points[0], self.original_points[1], k=2
            )
            poi_values = np.linspace(
                poi_values_original[0] - 2, poi_values_original[-1] + 2, self.n_interpol
            )
            logger.info(
                "Extrapolating between {} and {}".format(poi_values[0], poi_values[-1])
            )
            two_dnll = self.dNLL_func(poi_values)
            self.interpolated_points = np.array([poi_values, two_dnll])

            self.check_points_and_compute_all_uncertainties()

    def check_points_and_compute_all_uncertainties(self):
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
        (
            self.down68,
            self.up68,
            self.down68_unc,
            self.up68_unc,
        ) = self.compute_uncertainties(1.0)
        (
            self.down95,
            self.up95,
            self.down95_unc,
            self.up95_unc,
        ) = self.compute_uncertainties(3.84)
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
                f"The NLL curve does not seem to cross the horizontal line for level {level}. Try scanning a wider range of points for {self.poi}!"
            )
            logger.info("Setting up and down to minimum and uncertainties to 0.")
            downs = [self.minimum]
            ups = [self.minimum]
            down_uncertainties = [0.0]
            up_uncertainties = [0.0]
        else:
            # even numbers are down, odd numbers are up
            down_idx = indices[::2]
            up_idx = indices[1::2]
            downs = [self.interpolated_points[:, i] for i in down_idx]
            logger.debug(f"Downs: {downs}")
            ups = [self.interpolated_points[:, i] for i in up_idx]
            logger.debug(f"Ups: {ups}")
            down_uncertainties = [abs(self.minimum[0] - down[0]) for down in downs]
            up_uncertainties = [abs(self.minimum[0] - up[0]) for up in ups]

        return downs, ups, down_uncertainties, up_uncertainties

    def cut_from_string(self, cut_string):
        """
        Cut the original points from the original points array using the cut_string
        :param cut_string: string with the cut
        :return: numpy array with the points that pass the cut
        """
        logger.debug(f"Cutting points with cut string: {cut_string}")
        if ">" in cut_string:
            thr = float(cut_string.split(">")[-1])
            if cut_string.startswith("0"):
                self.original_points = self.original_points[
                    :, self.original_points[0] > thr
                ]
            if cut_string.startswith("1"):
                self.original_points = self.original_points[
                    :, self.original_points[1] > thr
                ]
        elif "<" in cut_string:
            thr = float(cut_string.split("<")[-1])
            if cut_string.startswith("0"):
                self.original_points = self.original_points[
                    :, self.original_points[0] < thr
                ]
            if cut_string.startswith("1"):
                self.original_points = self.original_points[
                    :, self.original_points[1] < thr
                ]

    def plot(self, ax, color=None, label=None, ylim=8.0, minimum_vertical_line=True):
        if label is None:
            label = self.poi
        # Restrict the plotted values to a dnll less than ylim
        x = self.interpolated_points[0][self.interpolated_points[1] < ylim]
        y = self.interpolated_points[1][self.interpolated_points[1] < ylim]
        ax.plot(x, y, color="k")
        # Vertical line passing through the minimum
        # Only plot in the case of single minimum
        if len(self.down68) == 1 and minimum_vertical_line:
            ax.plot(
                [self.minimum[0], self.minimum[0]],
                [self.minimum[1], self.up68[0][1]],
                color=color,
                linestyle="--",
            )
        # Vertical line passing through down68
        for down68 in self.down68:
            ax.plot(
                [down68[0], down68[0]],
                [self.minimum[1], down68[1]],
                color="k",
                linestyle="--",
            )
        # Vertical line passing through up68
        for up68 in self.up68:
            ax.plot(
                [up68[0], up68[0]],
                [self.minimum[1], up68[1]],
                color="k",
                linestyle="--",
            )
        # Vertical line passing through down95
        for down95 in self.down95:
            ax.plot(
                [down95[0], down95[0]],
                [self.minimum[1], down95[1]],
                color="k",
                linestyle="--",
            )
        # Vertical line passing through up95
        for up95 in self.up95:
            ax.plot(
                [up95[0], up95[0]],
                [self.minimum[1], up95[1]],
                color="k",
                linestyle="--",
            )
        # Points where NLL crosses 1
        for i, (down68, up68) in enumerate(zip(self.down68, self.up68)):
            ax.plot(
                [down68[0], up68[0]],
                [down68[1], up68[1]],
                color=color,
                linestyle="",
                marker="o",
                label=label if i == 0 else None,
            )
        # Points where NLL crosses 4
        for down95, up95 in zip(self.down95, self.up95):
            ax.plot(
                [down95[0], up95[0]],
                [down95[1], up95[1]],
                color=color,
                linestyle="",
                marker="o",
            )

        return ax

    def plot_simple(self, ax, color="k", label=None, ylim=8.0, linestyle="-"):
        if label is None:
            label = self.poi
        # Restrict the plotted values to a dnll less than ylim
        x = self.interpolated_points[0][self.interpolated_points[1] < ylim]
        y = self.interpolated_points[1][self.interpolated_points[1] < ylim]
        logger.debug(f"min x: {x[0]}, max x: {x[-1]}")
        logger.debug(f"min y {min(y)}, max y {max(y)}")
        ax.plot(x, y, color=color, label=label, linewidth=2, linestyle=linestyle)

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

    def get_bestfit_string(self):
        if len(self.down68) > 1:
            return ""
        else:
            nomstring = f"{self.minimum[0]:.3f}"
            upstring = f"{self.up68_unc[0]:.3f}"
            upstring = "{+" + upstring + "}"
            downstring = f"{self.down68_unc[0]:.3f}"
            downstring = "{-" + downstring + "}"
            poi_string = get_parameter_label(self.poi)
            return f"{poi_string} = ${nomstring}^{upstring}_{downstring}$"

    def get_68interval_string(self):
        if len(self.down68) > 1:
            return ""
        else:
            poi_string = get_parameter_label(self.poi)
            return f"{poi_string}: [{self.down68[0][0]:.3f}, {self.up68[0][0]:.3f}] (68% CL)"
    
    def get_95interval_string(self):
        if len(self.down95) > 1:
            return ""
        else:
            poi_string = get_parameter_label(self.poi)
            return f"{poi_string}: [{self.down95[0][0]:.3f}, {self.up95[0][0]:.3f}] (95% CL)"


class ScanSingles:
    def __init__(
        self,
        poi,
        input_dirs,
        skip_best=False,
        file_name_tmpl=None,
        cut_strings=None,
        allow_extrapolation=True,
    ):  # skip_best is useless here, but to keep the interface the same
        self.file_name_tmpl = "higgsCombine_SINGLES_{}".format(poi)
        self.tree_name = "limit"
        self.poi = poi
        file_name = [
            f for f in os.listdir(input_dirs[0]) if f.startswith(self.file_name_tmpl)
        ][0]

        f = uproot.open(os.path.join(input_dirs[0], file_name))
        logger.info(f"Opened file {file_name} for SINGLES scan")
        t = f[self.tree_name]
        arr = t.arrays()
        logger.info("Found following array for singles: {}".format(arr[self.poi]))
        self.mu = arr[self.poi][0]
        self.mu_down = arr[self.poi][1]
        self.mu_up = arr[self.poi][2]


class Scan2D:
    def __init__(
        self,
        pois,
        file_name_template,
        input_dirs,
        skip_best=False,
        best_fit_file=None,
        model_config=None,
        specs_name=None,
    ):
        if pois is None:
            raise ValueError("pois must be a list of strings")
        self.pois = pois
        self.file_name_tmpl = file_name_template
        self.default_cut = "deltaNLL<990.0"
        self.tree_name = "limit"

        logger.info(f"Looking for files {file_name_template} inside {input_dirs}")
        logger.debug(f"POIs: {self.pois}")

        dirs_template = [
            "{}/{}:{}".format(input_dir, self.file_name_tmpl, self.tree_name)
            for input_dir in input_dirs
        ]

        if skip_best:
            to_concatenate = []
            if uproot_main_version > 4:
                files = []
                for dr in dirs_template:
                    dr = dr.split(":")[0]
                    files += glob.glob(dr)
                for f in files:
                    f = uproot.open(f)
                    t = f[self.tree_name]
                    batch = t.arrays([*self.pois, "deltaNLL"])
                    to_concatenate.append(batch[1:])
            else:
                for batch in uproot.iterate(dirs_template, cut=self.default_cut):
                    batch = batch[[*self.pois, "deltaNLL"]]
                    to_concatenate.append(batch[1:])
            branches = ak.concatenate(to_concatenate)
            branches = ak.to_numpy(branches)
        else:
            branches = uproot.concatenate(
                dirs_template,
                expressions=[*self.pois, "deltaNLL"],
                cut=self.default_cut,
                library="np",
            )

        x = branches[pois[0]]
        y = branches[pois[1]]
        z = 2 * branches["deltaNLL"]

        # remove values for which z < -100
        # this was added on 21.08.23 because one of the stupid TK fits was going bananas at the corners
        # due to two single failed points out of more than 4000
        # I spotted it was a failed point because the z value was -268435456.000
        mask = z > -1000
        x = x[mask]
        y = y[mask]
        z = z[mask]

        # apply extra selections via specs_name if present
        #if specs_name == "top_floatingBR_ctcg_HggHZZHttHttBoostHbbVBF_asimov":
        #    logger.info("Applying extra selections for top_floatingBR_ctcg_HggHZZHttHttBoostHbbVBF_asimov")
        #    # remove points  inside a square with x in [2.4, 2.8] and y in [0.11, 0.14]
        #    mask = ~np.logical_and(
        #        np.logical_and(x > 2.4, x < 2.8), np.logical_and(y > 0.11, y < 0.14)
        #    )
        #    x = x[mask]
        #    y = y[mask]
        #    z = z[mask]
        #if specs_name == "top_floatingBR_ctcg_HggHZZHttHttBoostHbbVBF":

        if specs_name == "top_floatingBR_ctcb_HggHZZHttHttBoostHbbVBF":
            logger.info("Applying extra selections for top_floatingBR_ctcb_HggHZZHttHttBoostHbbVBF")
            mask = ~np.logical_and(
                np.logical_and(x > 0.7, x < 1.2), np.logical_and(y > -7, y < -5.2)
            )
            x = x[mask]
            y = y[mask]
            z = z[mask]
            mask = ~np.logical_and(
                np.logical_and(x > -2.2, x < -1.2), np.logical_and(y > -11, y < -7)
            )
            x = x[mask]
            y = y[mask]
            z = z[mask]
       
        # Sanity check: min and max of x and y of the found files
        logger.debug(f"Sanity check: min x: {min(x)}, max x: {max(x)}")
        logger.debug(f"Sanity check: min y: {min(y)}, max y: {max(y)}")
        logger.debug(f"Sanity check: min z: {min(z)}, max z: {max(z)}")

        self.points = np.array([x, y, z])
        # Remove nans
        self.points = self.points[:, ~np.isnan(self.points).any(axis=0)]

        x, y, z = self.points
        # dump points to debug
        import pickle
        output_file = "/work/gallim/DifferentialCombination_home/DiffCombOrchestrator/tries/debug_plot2D/points.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(self.points, f)

        logger.debug(
            f"Points:\nx: {list(self.points[0])}\ny: {list(self.points[1])}\nz: {list(self.points[2])}"
        )
        
        #z -= z.min()
        logger.debug("z after subtracting minimum: {}".format(list(z)))

        self.minimum = self.points[:, np.argmin(z)]

        if best_fit_file is not None:
            logger.info(f"Using best fit from file {best_fit_file}")
            f = uproot.open(best_fit_file)
            t = f[self.tree_name]
            arr = t.arrays([self.pois[0], self.pois[1]])
            self.minimum = np.array([arr[self.pois[0]][0], arr[self.pois[1]][0]])

        if model_config:
            x_min = np.min([model_config[pois[0]][0], x.min()])
            x_max = np.max([model_config[pois[0]][1], x.max()])
            y_min = np.min([model_config[pois[1]][0], y.min()])
            y_max = np.max([model_config[pois[1]][1], y.max()])
        else:
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()

        logger.debug(f"Interpolating points between {x_min} and {x_max} for {pois[0]}")
        logger.debug(f"Interpolating points between {y_min} and {y_max} for {pois[1]}")
        # self.y_int, self.x_int = np.mgrid[y_min:y_max:1000j, x_min:x_max:1000j]
        
        self.y_int, self.x_int = np.mgrid[y_min:y_max:500j, x_min:x_max:500j]
        self.z_int = griddata(
            (x, y), z, (self.x_int, self.y_int), method="cubic", fill_value=10.0
            #(x, y), z, (self.x_int, self.y_int), method="cubic", fill_value=20.0
        )
        #self.x_int, self.y_int, self.z_int = np.meshgrid(
        #    x, y, z
        #)
        #self.x_int = self.x_int[:, :, 0]
        #self.y_int = self.y_int[:, :, 0]
        #self.z_int = self.z_int[:, :, 0]
        #print(self.x_int.shape)
        #print(self.y_int.shape)
        #print(self.z_int.shape)

        self.z_int[0] -= self.z_int.min()
        self.z_int[1] -= self.z_int.min()

        # smooth the z_int
        self.z_int = gaussian_filter(self.z_int, sigma=5.0)

    def plot_as_heatmap(self, ax):
        colormap = custom_colormap("Purples")
        pc = ax.pcolormesh(
            self.x_int,
            self.y_int,
            self.z_int,
            vmin=0,
            vmax=10,
            cmap=colormap,
            shading="gouraud",
        )

        return ax, colormap, pc

    def plot_as_colorful_heatmap(self, ax):
        colormap = plt.get_cmap("plasma")
        colormap = truncate_colormap(colormap, 0.2, 1.0, 1000)
        pc = ax.pcolormesh(
            self.x_int,
            self.y_int,
            self.z_int,
            vmin=0,
            vmax=10,
            cmap=colormap,
            shading="gouraud",
        )

        return ax, colormap, pc

    def plot_as_contour(self, ax, color="k", label=None):
        cs = ax.contour(
            self.x_int,
            self.y_int,
            self.z_int,
            levels=[2.295748928898636, 6.180074306244173],
            colors=[color, color],
            linewidths=[2.2, 2.2],
            linestyles=["solid", "dashed"],
            #levels=[2.295748928898636, 6.180074306244173, 11.829158081900795, 19.333908611934685],
            #colors=[color, color, color, color],
            #linewidths=[2.2, 2.2, 2.2, 2.2],
            #linestyles=["solid", "dashed", "dashed", "dashed"],

        )
        # add labels
        try:
            levels = ["68%", "95%"]
            #levels = ["68%", "95%", "99.7%", "99.9%"]
            if label is not None:
                for i, cl in enumerate(levels):
                    cs.collections[i].set_label(f"{label} {cl}")
        except IndexError as e:
            logger.warning(
                f"Could not add labels to contour plot because of error {e}."
            )
            if label is not None:
                cs.collections[0].set_label(f"{label} 68%")

        # Best value as point
        ax.plot(
            [self.minimum[0]],
            [self.minimum[1]],
            color=color,
            linestyle="",
            marker="o",
            label=f"{label} best fit",
        )

        return ax

    def plot_as_contourf(self, ax):
        cs = ax.contourf(
            self.x_int,
            self.y_int,
            self.z_int,
            levels=[0.0, 2.295748928898636, 6.180074306244173],
            # colors=["darkorange", "yellow"],
            colors=["navy", "cornflowerblue"],
        )

        return ax
