""" Script to plot NLLs and final (superimposed) plots for a single variable.
Following are the main parameters:
- 
"""
import argparse
import os

# Needed to fix the fucking 
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
matplotlib.use("AGG")

from differential_combination_postprocess.utils import setup_logging, extract_from_yaml_file
from differential_combination_postprocess.scan import Scan, DifferentialSpectrum
from differential_combination_postprocess.figures import XSNLLsPerPOI, DiffXSsPerObservable


def parse_arguments():
    parser = argparse.ArgumentParser(
            description="Produce XS plots"
            )

    parser.add_argument(
        "--level",
        type=str,
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Level of logger"
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
    logger = setup_logging(args.level)
    variable = args.variable
    input_dir = args.input_dir
    metadata_dir = args.metadata_dir
    output_dir = args.output_dir
    categories = args.categories
    logger.info("Plotting session for variable {}".format(variable))

    # First produce NLL plots, one for each category
    # Each containing the NLL curves for each POI
    logger.info("Working with the following categories: {}".format(categories))
    categories_yamls = ["{}.yml".format(category) for category in categories]
    logger.debug("YAMLs: {}".format(categories_yamls))

    differential_spectra = {}

    for fl in categories_yamls:
        full_path_to_file = "{}/{}/{}".format(metadata_dir, variable, fl)
        logger.debug("Full path to file: {}".format(full_path_to_file))
        # Based on the assumption that we have a config file for each category called 'category_name.yaml'
        category = fl.split(".")[0]
        metadata_dct = extract_from_yaml_file(full_path_to_file)
        categories_numbers = [directory for directory in os.listdir(input_dir) if category in directory]
        logger.debug("Will look into the following directories: {}".format(categories_numbers))
        category_input_dirs = ["{}/{}".format(input_dir, directory) for directory in categories_numbers]

        # metadata_dct has the format {"yield_par": "poi"}
        logger.debug("metadata_dct: {}".format(metadata_dct))
        pois = list(set(metadata_dct.values()))

        diff_spectrum = DifferentialSpectrum(variable, category, pois, category_input_dirs)
        differential_spectra[category] = diff_spectrum

        plot_to_dump = XSNLLsPerPOI(diff_spectrum)
        plot_to_dump.dump(output_dir)

    # Produce the final differential xs plot including all the categories
    shapes = {}
    for category in categories:
        shapes[category] = {}

    from differential_combination_postprocess.shapes import njets_obs_shape 
    final_plot = DiffXSsPerObservable([njets_obs_shape])
    final_plot.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)