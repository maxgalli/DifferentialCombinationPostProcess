from tokenize import group
from differential_combination_postprocess.scan import DifferentialSpectrum
from differential_combination_postprocess.figures import XSNLLsPerCategory, NScans
from differential_combination_postprocess.utils import setup_logging
from differential_combination_postprocess.scan import Scan2D
from differential_combination_postprocess.figures import TwoDScansPerModel

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

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where output files will be stored",
    )

    parser.add_argument(
        "--other-input-dir",
        type=str,
        required=False,
        help="Other directory where the .root files with 'limit' trees are stored",
    )

    parser.add_argument(
        "--skip-best", action="store_true", help="Skip printing the best point"
    )

    parser.add_argument(
        "--file-name-tmpl",
        type=str,
        required=False,
        default=None,
        help="File name template in case of 2D scans to plot the 1D ones, since the convention is different from the combination ones",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--poi", type=str, help="POI to plot in single scan")
    group.add_argument("--pois", nargs="+", help="POIs to plot in 2D")

    return parser.parse_args()


def main(args):
    logger = setup_logging(level="DEBUG")

    if args.poi:
        logger.info("Plotting single scan for {}".format(args.poi))
        ds = DifferentialSpectrum(
            "POI", "category", [args.poi], [args.input_dir], skip_best=args.skip_best
        )

        fig = XSNLLsPerCategory(ds, ylim=8, print_best=True)
        fig.output_name = f"xs_scan_{args.poi}"
        fig.dump(args.output_dir)

        # Plot original points
        logger.info("Plotting original points")
        f, ax = plt.subplots(figsize=(8, 6))
        x = ds.scans[args.poi].original_points[0]
        y = ds.scans[args.poi].original_points[1] / 2
        ax.scatter(x, y, marker="o", color="black", s=10)
        f.savefig(f"{args.output_dir}/{args.poi}_original_points.pdf")
        f.savefig(f"{args.output_dir}/{args.poi}_original_points.png")

        # Plot 2nll
        logger.info("Plotting 2nll")
        f, ax = plt.subplots(figsize=(8, 6))
        x2 = ds.scans[args.poi].original_points[0]
        y2 = ds.scans[args.poi].original_points[1]
        ax.scatter(x2, y2, marker="o", color="black", s=10)
        f.savefig(f"{args.output_dir}/{args.poi}_2nll.pdf")
        f.savefig(f"{args.output_dir}/{args.poi}_2nll.png")

        if args.other_input_dir:
            logger.info("Plotting other scan")
            ds_other = DifferentialSpectrum(
                "POI",
                "category",
                [args.poi],
                [args.other_input_dir],
                skip_best=args.skip_best,
            )

            fig = NScans(
                {
                    f"{args.poi}_main": ds.scans[args.poi],
                    f"{args.poi}_other": ds_other.scans[args.poi],
                }
            )
            fig.dump(args.output_dir)

            f, ax = plt.subplots(figsize=(8, 6))
            x_other = ds_other.scans[args.poi].original_points[0]
            y_other = ds_other.scans[args.poi].original_points[1] / 2
            ax.scatter(x, y, marker="o", color="black", s=10, label=args.input_dir)
            ax.scatter(
                x_other,
                y_other,
                marker="p",
                color="red",
                s=10,
                label=args.other_input_dir,
            )
            ax.legend(prop={"size": 4})
            f.savefig(f"{args.output_dir}/{args.poi}_other_points.pdf")
            f.savefig(f"{args.output_dir}/{args.poi}_other_points.png")

            f, ax = plt.subplots(figsize=(8, 6))
            x2_other = ds_other.scans[args.poi].original_points[0]
            y2_other = ds_other.scans[args.poi].original_points[1]
            ax.scatter(x2, y2, marker="o", color="black", s=10, label=args.input_dir)
            ax.scatter(
                x2_other,
                y2_other,
                marker="p",
                color="red",
                s=10,
                label=args.other_input_dir,
            )
            ax.legend(prop={"size": 4})
            f.savefig(f"{args.output_dir}/{args.poi}_other_2nll.pdf")
            f.savefig(f"{args.output_dir}/{args.poi}_other_2nll.png")

    if args.pois:
        logger.info("Plotting 2D scan for {}".format(args.pois))
        tmpl = "higgsCombine_SCAN_*"
        scans = {}
        scans["test"] = Scan2D(args.pois, tmpl, [args.input_dir])
        plot = TwoDScansPerModel(scans, "test", "test")
        plot.dump(args.output_dir)

        if args.file_name_tmpl:
            for poi in args.pois:
                ds = DifferentialSpectrum(
                    "POI",
                    "category",
                    [poi],
                    [args.input_dir],
                    file_name_tmpl=args.file_name_tmpl,
                )
                fig = XSNLLsPerCategory(ds, ylim=8, print_best=True)
                fig.output_name = f"xs_scan_{poi}"
                fig.dump(args.output_dir)


if __name__ == "__main__":
    main(parse_arguments())
