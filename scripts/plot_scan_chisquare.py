import argparse
import os
import pickle

# Needed to fix the fucking
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("AGG")

from differential_combination_postprocess.utils import (
    setup_logging,
    extract_from_yaml_file,
)
from differential_combination_postprocess.scan import Scan, Scan2D
from differential_combination_postprocess.figures import ScanChiSquare


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot lin vs quad SMEFT scans")

    parser.add_argument("--scan-input-dir", type=str, required=True, help="")

    parser.add_argument(
        "--chi-file",
        type=str,
        required=False,
        help="Path to a pickle file containing the chi square result",
    )

    parser.add_argument("--output-dir", type=str, help="")

    parser.add_argument(
        "--scan-name-tmpl",
        type=str,
        required=False,
        default=None,
        help="File name templates",
    )
    parser.add_argument(
        "--chi-how",
        type=str,
        required=False,
        default="expected_profiled",
        choices=[
            "expected_profiled",
            "expected_fixed",
            "observed_profiled",
            "observed_fixed",
        ],
        help="expected_fixed, expected_profiled, observed_fixed, observed_profiled",
    )

    parser.add_argument("--poi", type=str, help="POI to plot in single scan")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    scan = Scan(
        poi=args.poi,
        input_dirs=[args.scan_input_dir],
        skip_best=True,
        file_name_tmpl=args.scan_name_tmpl,
        cut_strings=None,
        allow_extrapolation=False,
    )
    with open(args.chi_file, "rb") as f:
        full_dct = pickle.load(f)
    chi_dct = full_dct[args.chi_how][args.poi]

    logger.info("Plotting the two scans")
    fig = ScanChiSquare(args.poi, scan, chi_dct)
    fig.dump(args.output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
