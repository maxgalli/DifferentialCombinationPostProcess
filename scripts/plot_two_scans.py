import argparse
import os

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
from differential_combination_postprocess.figures import TwoScans


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot lin vs quad SMEFT scans")

    parser.add_argument("--input-dir", type=str, required=True, help="")

    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label to be used in the output file name",
    )

    parser.add_argument(
        "--other-input-dir",
        type=str,
        required=False,
        help="Other directory where the .root files with 'limit' trees are stored",
    )

    parser.add_argument(
        "--other-label",
        type=str,
        required=False,
        help="Label to be used in the output file name",
    )

    parser.add_argument("--output-dir", type=str, help="")

    parser.add_argument(
        "--file-name-tmpl",
        type=str,
        required=False,
        default=None,
        help="File name templates",
    )

    parser.add_argument("--poi", type=str, help="POI to plot in single scan")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    scans = {}
    for label, input_dir in zip(
        [args.label, args.other_label], [args.input_dir, args.other_input_dir]
    ):
        scans[label] = Scan(
            poi=args.poi,
            input_dirs=[input_dir],
            skip_best=True,
            file_name_tmpl=args.file_name_tmpl,
            cut_strings=None,
            allow_extrapolation=False,
        )

    logger.info("Plotting the two scans")
    fig = TwoScans(
        args.poi,
        "",
        scans[args.label],
        scans[args.other_label],
        args.label,
        args.other_label,
        "TwoScans",
    )
    fig.dump(args.output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
