import argparse
import os

# Needed to fix the fucking 
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
matplotlib.use("AGG")

from differential_combination_postprocess.utils import setup_logging, extract_from_yaml_file
from differential_combination_postprocess.scan import Scan, DifferentialSpectrum
from differential_combination_postprocess.figures import XSNLLsPerPOI


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
        "--single-category",
        type=str,
        help="Dump NLL plots for a single category instead of looping on all of them"
    )

    return parser.parse_args()


def main(args):
    logger = setup_logging(args.level)
    variable = args.variable
    input_dir = args.input_dir
    metadata_dir = args.metadata_dir
    output_dir = args.output_dir
    logger.info("Plotting session for variable {}".format(variable))

    # First produce NLL plots, one for each category
    # Each containing the NLL curves for each POI
    if args.single_category:
        logger.info("Producing NLL plot only for {}".format(args.single_category))
        categories_yamls = ["{}.yml".format(args.single_category)]
    else:
        categories_yamls = os.listdir(metadata_dir)
    for fl in categories_yamls:
        full_path_to_file = "{}/{}".format(metadata_dir, fl)
        # Based on the assumption that we have a config file for each category called 'category_name.yaml'
        category = fl.split(".")[0]
        metadata_dct = extract_from_yaml_file(full_path_to_file)
        categories_numbers = [directory for directory in os.listdir(input_dir) if category in directory]
        logger.debug("Will look into the following directories: {}".format(categories_numbers))
        category_input_dirs = ["{}/{}".format(input_dir, directory) for directory in categories_numbers]

        # metadata_dct has the format {"binning": {"poi1": [], "poi2": []}}
        pois = list(metadata_dct["binning"].keys())

        diff_spectrum = DifferentialSpectrum(variable, category, pois, category_input_dirs)

        plot_to_dump = XSNLLsPerPOI(diff_spectrum)
        plot_to_dump.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)