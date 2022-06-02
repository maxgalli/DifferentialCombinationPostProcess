""" Script to plot NLLs and final (superimposed) plots for a single observable.
Following are the main parameters:
- 
"""
import argparse
import os
import re
import numpy as np
from copy import deepcopy

# Needed to fix the fucking
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("AGG")

from differential_combination_postprocess.utils import (
    setup_logging,
    extract_from_yaml_file,
)
from differential_combination_postprocess.scan import Scan, DifferentialSpectrum
from differential_combination_postprocess.figures import (
    XSNLLsPerCategory,
    XSNLLsPerPOI,
    XSNLLsPerPOI_Full,
    DiffXSsPerObservable,
)
from differential_combination_postprocess.shapes import ObservableShapeFitted, sm_shapes
from differential_combination_postprocess.physics import analyses_edges, overflows

import logging


def parse_arguments():
    parser = argparse.ArgumentParser(description="Produce XS plots")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    parser.add_argument(
        "--observable",
        required=True,
        type=str,
        help="observable to produce XS plots for",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs",
        help="Directory where the .root files with 'limit' trees are stored",
    )

    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="outputs",
        help="Directory where the .yaml files with metadata are stored",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where output files will be stored",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        required=True,
        help="Categories for which NLL plots are dumped, along with the superimposed final ones",
    )

    parser.add_argument(
        "--systematic-bands",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="Categories for which we want to plot the systematics bands in the final plot",
    )

    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        type=str,
        default=[],
        help="Directory to exclude from the list of input directories",
    )

    parser.add_argument(
        "--no-final",
        action="store_true",
        default=False,
        help="Do not produce final plots",
    )

    parser.add_argument(
        "--no-nll",
        action="store_true",
        default=False,
        help="Skip production of NLL plots",
    )

    return parser.parse_args()


def get_shapes_from_differential_spectra(differential_spectra, observable):
    shapes = []
    for category, spectrum in differential_spectra.items():
        simple_category = category.split("_")[0]
        # Remember that the results we get for mu are meant to scale the SM cross section
        logging.info(
            f"Building shape for category {category}, observable {observable} and edges {analyses_edges[observable][simple_category]}"
        )
        # First: copy the finest possible shape (Hgg) and rebin it with what we need
        sm_rebinned_shape = deepcopy(sm_shapes[observable])
        logging.debug(f"SM original xs: {sm_rebinned_shape.xs}")
        logging.debug(
            f"SM original xs_over_binwidth: {sm_rebinned_shape.xs_over_bin_width}"
        )
        sm_rebinned_shape.rebin(analyses_edges[observable][simple_category])
        # Second: bin-wise multiply the new bin values by the values for mu computed in the scan
        mus = np.array([scan.minimum[0] for scan in spectrum.scans.values()])
        mus_up = np.array(
            [scan.minimum[0] + scan.up68_unc for scan in spectrum.scans.values()]
        )
        mus_down = np.array(
            [scan.minimum[0] - scan.down68_unc for scan in spectrum.scans.values()]
        )
        logging.debug(f"Ordered mus for category {category}: {mus}")
        logging.debug(f"SM rebinned xs: {sm_rebinned_shape.xs}")
        logging.debug(
            f"SM rebinned xs_over_binwidth: {sm_rebinned_shape.xs_over_bin_width}"
        )
        weighted_bins = np.multiply(np.array(sm_rebinned_shape.xs), mus)
        logging.debug(f"Reweighted xs: {weighted_bins}")
        weighted_bins_up = np.multiply(np.array(sm_rebinned_shape.xs), mus_up)
        weighted_bins_down = np.multiply(np.array(sm_rebinned_shape.xs), mus_down)
        logging.debug(f"Reweighted xs up: {weighted_bins_up}")
        logging.debug(f"Reweighted xs down: {weighted_bins_down}")
        shapes.append(
            ObservableShapeFitted(
                observable,
                simple_category,
                analyses_edges[observable][simple_category],
                weighted_bins,
                weighted_bins_up,
                weighted_bins_down,
                overflow=observable in overflows,
            )
        )
        logging.debug(f"Just appended shape {shapes[-1]}")

    return shapes


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    observable = args.observable
    input_dir = args.input_dir
    metadata_dir = args.metadata_dir
    output_dir = args.output_dir
    categories = args.categories
    systematic_bands = args.systematic_bands
    exclude_dirs = args.exclude_dirs
    logger.info(f"Plotting session for observable {observable}")

    # First produce NLL plots, one for each category
    # Each containing the NLL curves for each POI
    logger.info(f"Working with the following categories: {categories}")
    categories_yamls = ["{}.yml".format(category) for category in categories]
    logger.debug(f"YAMLs: {categories_yamls}")

    differential_spectra = {}
    differential_spectra_statonly = {}

    for fl in categories_yamls:
        full_path_to_file = f"{metadata_dir}/{observable}/{fl}"
        logger.debug(f"Full path to file: {full_path_to_file}")
        # Based on the assumption that we have a config file for each category called 'category_name.yaml'
        category = fl.split(".")[0]
        pois = extract_from_yaml_file(full_path_to_file)

        # Here define categories for asimov, statonly and asimov_statonly
        asimov_cat = f"{category}_asimov"
        statonly_cat = f"{category}_statonly"
        asimov_statonly_cat = f"{category}_asimov_statonly"

        sub_categories = [category, asimov_cat, statonly_cat, asimov_statonly_cat]
        if args.no_nll:
            sub_categories = [category, statonly_cat]

        # Plot scans for nominal, statonly, asimov and asimov_statonly
        sub_cat_spectra = {}
        for sub_cat in sub_categories:
            logger.info(f"Working on sub-category {sub_cat}")
            regex = re.compile(f"{sub_cat}(-[0-9])?$")
            categories_numbers = [
                directory
                for directory in os.listdir(input_dir)
                if regex.match(directory)
            ]
            # Give the possibility to exclude some directories like the ones for which jobs are still running
            categories_numbers = [
                directory
                for directory in categories_numbers
                if directory not in exclude_dirs
            ]
            if len(categories_numbers) == 0:
                logger.warning(f"No directories found for category {sub_cat}")
                continue
            logger.debug(
                f"Will look into the following directories: {categories_numbers}"
            )
            category_input_dirs = [
                f"{input_dir}/{directory}" for directory in categories_numbers
            ]

            diff_spectrum = DifferentialSpectrum(
                observable, sub_cat, pois, category_input_dirs
            )
            sub_cat_spectra[sub_cat] = diff_spectrum
            if sub_cat == category:
                differential_spectra[sub_cat] = diff_spectrum
            if sub_cat == statonly_cat and sub_cat.split("_")[0] in systematic_bands:
                differential_spectra_statonly[sub_cat] = diff_spectrum

            if not args.no_nll:
                plot_to_dump = XSNLLsPerCategory(diff_spectrum)
                plot_to_dump.dump(output_dir)
                plt.close("all")

        # Plot one figure per POI with nominal, statonly, asimov and asimov_statonly NLLs
        if not args.no_nll:
            logger.info(
                "Now plotting one figure per POI with nominal, statonly, asimov and asimov_statonly NLLs (if found)"
            )
            poi_plots = XSNLLsPerPOI(sub_cat_spectra, debug=args.debug)
            poi_plots.dump(output_dir)

        if args.debug and not args.no_nll:
            full_plot_to_dump = XSNLLsPerPOI_Full(sub_cat_spectra)
            full_plot_to_dump.dump(output_dir)

    if "inclusive" not in categories[0]:
        logger.debug(f"Differential spectra: {differential_spectra}")

        # Produce the final differential xs plot including all the categories
        logger.info(
            f"Now producing the final differential xs plot for observable {observable}"
        )

        shapes = get_shapes_from_differential_spectra(differential_spectra, observable)
        shapes_statonly = get_shapes_from_differential_spectra(
            differential_spectra_statonly, observable
        )
        # horrible
        # I should probably introduce another dict
        shapes_systonly = []
        for shape in shapes:
            for shape_statonly in shapes_statonly:
                if (
                    shape.category == shape_statonly.category
                    and shape.observable == shape_statonly.observable
                ):
                    shapes_systonly.append(shape - shape_statonly)
        if args.debug:
            for shape in shapes_systonly:
                logger.debug(f"Systematic shape: \n{shape}")

        if not args.no_final:
            final_plot_output_name = f"Final-{observable}-" + "_".join(categories)
            final_plot = DiffXSsPerObservable(
                final_plot_output_name, sm_shapes[observable], shapes, shapes_systonly
            )
            final_plot.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
