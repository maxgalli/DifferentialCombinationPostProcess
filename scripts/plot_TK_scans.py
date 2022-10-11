"""
Run with e.g.:
plot_TK_scans.py --model yukawa_coupdep --input-dir /work/gallim/DifferentialCombination_home/DifferentialCombinationRun2/outputs/TK_scans/Yukawa_NOTscalingbbH_couplingdependentBRs --output-dir /eos/home-g/gallim/www/plots/DifferentialCombination/CombinationRun2/TK_plots --categories HggHZZHWWHtt --combination HggHZZHWWHtt --expected
"""
import argparse
import os

# Needed to fix the fucking
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("AGG")

from differential_combination_postprocess.utils import setup_logging
from differential_combination_postprocess.figures import TwoDScansPerModel
from differential_combination_postprocess.scan import Scan2D
from differential_combination_postprocess.physics import TK_models as models


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot 2D TK fits")

    parser.add_argument(
        "--model",
        required=True,
        choices=list(models.keys()),
        help="Model used for the fit",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="inputs",
        help="Directory where the .root files with 'limit' trees are stored",
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
        required=False,
        default=[],
        help="Categories for which the plots are produced",
    )

    parser.add_argument(
        "--combination",
        type=str,
        help="Which of the categories is printed as colored map instead of lines",
    )

    parser.add_argument(
        "--expected", action="store_true", help="Look for and plot asimov results"
    )

    parser.add_argument(
        "--expected-bkg",
        action="store_true",
        help="When plotting observed values, plot the expected 2NLL for the combination instead as coloured heatmap",
    )

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    input_dir = args.input_dir
    output_dir = args.output_dir
    categories = args.categories
    combination = args.combination

    file_name_template = "higgsCombine_SCAN_*"
    # infer pois from model, order is important
    pois = list(models[args.model].keys())

    logger.info(f"Working with the following categories: {categories}")
    logger.info(f"Will use {combination} as combination category")

    scan_dict = {}
    for category in categories:
        if args.expected:
            category += "_asimov"
        input_subdirs = [
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if d.startswith(f"{category}-")
        ]
        best_fit_file = os.path.join(
            input_subdirs[0],
            f"higgsCombine_POSTFIT_{category}.MultiDimFit.mH125.38.root",
        )
        if args.expected:
            best_fit_file = os.path.join(
                input_subdirs[0], "higgsCombineAsimovBestFit.MultiDimFit.mH125.38.root"
            )
        scan_dict[category] = Scan2D(
            pois,
            file_name_template,
            input_subdirs,
            skip_best=True,
            best_fit_file=best_fit_file,
            model_config=models[args.model],
        )
    logger.debug(f"Scan dictionary: {scan_dict}")

    # if we want the expected as bkg, make the scan
    expected_combination_scan = None
    if args.expected_bkg:
        category = f"{combination}_asimov"
        input_subdirs = [
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if d.startswith(category)
        ]
        best_fit_file = os.path.join(
            input_subdirs[0], "higgsCombineAsimovBestFit.MultiDimFit.mH125.38.root"
        )
        expected_combination_scan = Scan2D(
            pois,
            file_name_template,
            input_subdirs,
            skip_best=True,
            best_fit_file=best_fit_file,
            model_config=models[args.model],
        )

    cat_names_string = "_".join(list(scan_dict.keys()))
    output_name = f"{args.model}_{cat_names_string}"
    if args.expected_bkg:
        output_name += "_expected_bkg"
    plot = TwoDScansPerModel(
        scan_dict=scan_dict,
        combination_name=combination + "_asimov"
        if args.expected_bkg or args.expected
        else combination,
        model_config=models[args.model],
        combination_asimov_scan=expected_combination_scan,
        output_name=output_name,
        is_asimov=args.expected,
    )
    plot.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
