""" Script to plot NLLs and final (superimposed) plots for a single variable.
Following are the main parameters:
- 
"""
import argparse
import os
import re
import numpy as np

# Needed to fix the fucking 
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
matplotlib.use("AGG")

from differential_combination_postprocess.utils import setup_logging, extract_from_yaml_file
from differential_combination_postprocess.scan import Scan, DifferentialSpectrum
from differential_combination_postprocess.figures import XSNLLsPerCategory, XSNLLsPerPOI, DiffXSsPerObservable
from differential_combination_postprocess.shapes import ObservableShapeFitted
from differential_combination_postprocess.physics import analyses_edges


def parse_arguments():
    parser = argparse.ArgumentParser(
            description="Produce XS plots"
            )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug messages"
    )

    parser.add_argument(
            "--variable",
            required=True,
            type=str,
            help="Variable to produce XS plots for"
            )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs",
        help="Directory where the .root files with 'limit' trees are stored"
    )

    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="outputs",
        help="Directory where the .yaml files with metadata are stored"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where output files will be stored"
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        required=True,
        help="Categories for which NLL plots are dumped, along with the superimposed final ones"
    )

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")
   
    variable = args.variable
    input_dir = args.input_dir
    metadata_dir = args.metadata_dir
    output_dir = args.output_dir
    categories = args.categories
    logger.info(f"Plotting session for variable {variable}")

    # First produce NLL plots, one for each category
    # Each containing the NLL curves for each POI
    logger.info(f"Working with the following categories: {categories}")
    categories_yamls = ["{}.yml".format(category) for category in categories]
    logger.debug(f"YAMLs: {categories_yamls}")

    differential_spectra = {}

    for fl in categories_yamls:
        full_path_to_file = f"{metadata_dir}/{variable}/{fl}"
        logger.debug(f"Full path to file: {full_path_to_file}")
        # Based on the assumption that we have a config file for each category called 'category_name.yaml'
        category = fl.split(".")[0]
        pois = extract_from_yaml_file(full_path_to_file)

        # Here define categories for asimov, statonly and asimov_statonly
        asimov_cat = f"{category}_asimov"
        statonly_cat = f"{category}_statonly"
        asimov_statonly_cat = f"{category}_asimov_statonly"

        # Plot scans for nominal and statonly
        sub_cat_spectra = {}
        for sub_cat in [category, statonly_cat, asimov_cat, asimov_statonly_cat]:
            logger.info(f"Working on sub-category {sub_cat}")
            regex = re.compile(f"{sub_cat}(-[0-9])?$")
            categories_numbers = [directory for directory in os.listdir(input_dir) if regex.match(directory)]
            if len(categories_numbers) == 0:
                logger.warning(f"No directories found for category {sub_cat}")
                continue
            logger.debug(f"Will look into the following directories: {categories_numbers}")
            category_input_dirs = [f"{input_dir}/{directory}" for directory in categories_numbers]

            diff_spectrum = DifferentialSpectrum(variable, sub_cat, pois, category_input_dirs)
            sub_cat_spectra[sub_cat] = diff_spectrum
            if sub_cat == category:
                differential_spectra[sub_cat] = diff_spectrum

            plot_to_dump = XSNLLsPerCategory(diff_spectrum)
            plot_to_dump.dump(output_dir)

        # Plot one figure per POI with nominal and statonly NLLs
        logger.info("Now plotting one figure per POI with nominal and statonly NLLs")
        poi_plots = XSNLLsPerPOI(sub_cat_spectra)
        poi_plots.dump(output_dir)


    # Produce the final differential xs plot including all the categories
    logger.info(f"Now producing the final differential xs plot for observable {variable}")
    shapes = []
    for category, spectrum in differential_spectra.items():
        logger.debug(f"Building shape for category {category}, variable {variable} and edges {analyses_edges[variable][category]}")
        shapes.append(
            ObservableShapeFitted(
                variable,
                analyses_edges[variable][category],
                np.array([scan.minimum[0] for scan in spectrum.scans.values()]),
                np.array([scan.minimum[0] + scan.up_uncertainty for scan in spectrum.scans.values()]),
                np.array([scan.minimum[0] - scan.down_uncertainty for scan in spectrum.scans.values()]),
            )
        )

    from differential_combination_postprocess.shapes import sm_shapes 
    final_plot = DiffXSsPerObservable(sm_shapes[variable], shapes)
    final_plot.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)