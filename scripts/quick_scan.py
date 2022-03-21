from differential_combination_postprocess.scan import DifferentialSpectrum
from differential_combination_postprocess.figures import XSNLLsPerCategory
from differential_combination_postprocess.utils import setup_logging

import argparse
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot quick scan")

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory where the .root files with 'limit' trees are stored",
    )

    parser.add_argument("--poi", type=str, required=True, help="POI to plot")

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where output files will be stored",
    )

    return parser.parse_args()


def main(args):
    logger = setup_logging(level="DEBUG")
    ds = DifferentialSpectrum("observable", "category", [args.poi], [args.input_dir])

    fig = XSNLLsPerCategory(ds)
    fig.output_name = f"xs_scan_{args.poi}"
    fig.dump(args.output_dir)

    # Plot original points
    logger.info("Plotting original points")
    f, ax = plt.subplots(figsize=(8, 6))
    x = ds.scans[args.poi].original_points[0]
    y = ds.scans[args.poi].original_points[1]
    ax.scatter(x, y, marker="o", color="black", s=10)
    f.savefig(f"{args.output_dir}/{args.poi}_original_points.pdf")
    f.savefig(f"{args.output_dir}/{args.poi}_original_points.png")

    # Plot 2nll
    logger.info("Plotting 2nll")
    f, ax = plt.subplots(figsize=(8, 6))
    x = ds.scans[args.poi].original_points[0]
    y = 2 * ds.scans[args.poi].original_points[1]
    ax.scatter(x, y, marker="o", color="black", s=10)
    f.savefig(f"{args.output_dir}/{args.poi}_2nll.pdf")
    f.savefig(f"{args.output_dir}/{args.poi}_2nll.png")


if __name__ == "__main__":
    main(parse_arguments())
