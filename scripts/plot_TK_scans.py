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
import numpy as np

matplotlib.use("AGG")

from differential_combination_postprocess.utils import setup_logging
from differential_combination_postprocess.figures import (
    TwoDScansPerModel,
    GenericNLLsPerPOI,
    TwoDScanDebug
)
from differential_combination_postprocess.scan import Scan2D, Scan
from differential_combination_postprocess.physics import TK_models as models

from plot_SMEFT_scans import plot_original_points


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
        "--statonly", action="store_true", help="Look for and plot statonly results"
    )

    parser.add_argument(
        "--expected-bkg",
        action="store_true",
        help="When plotting observed values, plot the expected 2NLL for the combination instead as coloured heatmap",
    )

    parser.add_argument("--twod-only", action="store_true", help="Only plot 2D scans")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    parser.add_argument("--debug-plots", action="store_true", help="Print debug plots with square points")

    return parser.parse_args()


oned_extra_selections = {
    "yukawa_coupdep_HggHZZHtt_asimov": {
        "kappab": lambda pois_values_original: ~np.logical_or(
            #np.logical_and(pois_values_original > -0.35, pois_values_original < -0.15),
            np.logical_and(pois_values_original > -0.7, pois_values_original < -0.15),
            np.logical_and(pois_values_original > 0.2, pois_values_original < 0.4),
            #np.logical_and(pois_values_original > 0.8, pois_values_original < 0.9)
        ),
        "kappac": lambda pois_values_original: ~np.logical_and(pois_values_original > -2., pois_values_original < -1.5)
    },
    "yukawa_coupdep_HggHZZHtt": {
        "kappab": lambda pois_values_original: ~np.logical_or(
            np.logical_and(pois_values_original > -1.4, pois_values_original < -1.15),
            np.logical_and(pois_values_original > 1.15, pois_values_original < 1.3)
        ),
        "kappac": lambda pois_values_original: ~np.logical_or(
            np.logical_and(pois_values_original > -4, pois_values_original < -3.5),
            np.logical_and(pois_values_original > 3.5, pois_values_original < 4)
        ),
    },
    "yukawa_floatingBR_HggHZZHtt_asimov": {
        "kappac": lambda pois_values_original: ~np.logical_and(pois_values_original > -1., pois_values_original < 1.),
        "kappab": lambda pois_values_original: ~np.logical_or(
            np.logical_and(pois_values_original > -0.1, pois_values_original < 0.3),
            np.logical_and(pois_values_original > 0.9, pois_values_original < 1.2)
        )
    },
}


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    input_dir = args.input_dir
    output_dir = args.output_dir
    categories = args.categories
    combination = args.combination

    # infer pois from model, order is important
    pois = list(models[args.model].keys())
    file_name_template_2d = f"higgsCombine_SCAN_2D{pois[0]}-{pois[1]}.*.root"
    # file_name_template_2d = "higgsCombine_SCAN_*"  # for old version

    subcat = "observed"
    subcat_suff = ""
    if args.expected:
        subcat = "expected"
        subcat_suff = "_asimov"
    if args.statonly:
        subcat += "_statonly"
        subcat_suff += "_statonly"

    logger.info(f"Working with the following categories: {categories}")
    logger.info(f"Will use {combination} as combination category")

    if not args.twod_only:
        logger.info("Plotting 1D scans")
        for coeff in pois:
            scans = {}
            for category in args.categories:
                input_dirs = [
                    os.path.join(input_dir, d)
                    for d in os.listdir(input_dir)
                    if d.startswith(f"{category}{subcat_suff}-")
                ]
                # get extra selections if present
                extra_selection = None
                if "{}_{}{}".format(args.model, category, subcat_suff) in oned_extra_selections:
                    if coeff in oned_extra_selections["{}_{}{}".format(args.model, category, subcat_suff)]:
                        extra_selection = oned_extra_selections["{}_{}{}".format(args.model, category, subcat_suff)][coeff]
                if len(input_dirs) == 0:
                    logger.warning(
                        f"No input directories found for {category}{subcat_suff}"
                    )
                    continue
                scans[category] = Scan(
                    coeff,
                    input_dirs,
                    skip_best=True,
                    file_name_tmpl=f"higgsCombine_SCAN_1D{coeff}.*.root",
                    extra_selections=extra_selection,
                )
                # All this should beput together smartly with the part before
                if args.expected_bkg:
                    logger.info("Plotting expected bkg for 1D scans")
                    category += "_asimov"
                    extra_selection = None
                    if "{}_{}{}".format(args.model, category, subcat_suff) in oned_extra_selections:
                        if coeff in oned_extra_selections["{}_{}{}".format(args.model, category, subcat_suff)]:
                            extra_selection = oned_extra_selections["{}_{}{}".format(args.model, category, subcat_suff)][coeff]
                    input_dirs = [
                        os.path.join(input_dir, d)
                        for d in os.listdir(input_dir)
                        if d.startswith(f"{category}-")
                    ]
                    if len(input_dirs) == 0:
                        logger.warning(
                            f"No input directories found for {category}"
                        )
                        continue
                    scans[category] = Scan(
                        coeff,
                        input_dirs,
                        skip_best=True,
                        file_name_tmpl=f"higgsCombine_SCAN_1D{coeff}.*.root",
                        extra_selections=extra_selection,
                    )
            if len(scans) > 0:
                fig = GenericNLLsPerPOI(coeff, scans, subcat, simple=True, plot_string=False, plot_interval=True, minimum_vertical_line=False)
                fig.dump(output_dir)
                fig = GenericNLLsPerPOI(
                    coeff, scans, subcat, simple=True, full_range=True
                )
                fig.dump(output_dir)
                logger.debug("Dumping original points")
                for scan_name, scan in scans.items():
                    plot_original_points(coeff, scan_name, scan, subcat, output_dir)

    # 2D scans
    scan_dict = {}
    for category in categories:
        if args.expected:
            category += "_asimov"
        if args.statonly:
            category += "_statonly"
        input_subdirs = [
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if d.startswith(f"{category}-")
        ]
        # order subdirs alphabetically
        input_subdirs.sort()
        if args.expected and "Hgg" in category:
            best_fit_file = os.path.join(
                input_subdirs[0], "higgsCombineAsimovBestFit.MultiDimFit.mH125.38.root"
            )
        else:
            best_fit_file = os.path.join(
                input_subdirs[0],
                [
                    f
                    for f in os.listdir(input_subdirs[0])
                    if f.startswith("higgsCombine_POSTFIT_")
                ][0],
            )
        scan_dict[category] = Scan2D(
            pois,
            file_name_template_2d,
            input_subdirs,
            skip_best=True,
            best_fit_file=best_fit_file,
            model_config=models[args.model],
            specs_name=f"{args.model}_{category}",
        )
    logger.debug(f"Scan dictionary: {scan_dict}")
    if args.debug_plots:
        for category, scan in scan_dict.items():
            logger.debug(f"Scan for {category}: {scan}")
            plot = TwoDScanDebug(scan, category)
            plot.dump(output_dir)

    # if we want the expected as bkg, make the scan
    expected_combination_scan = None
    if args.expected_bkg:
        category = f"{combination}_asimov"
        if args.statonly:
            category += "_statonly"
        input_subdirs = [
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if d.startswith(category)
        ]
        # order subdirs alphabetically
        input_subdirs.sort()
        best_fit_file = os.path.join(
            input_subdirs[0], "higgsCombineAsimovBestFit.MultiDimFit.mH125.38.root"
        )
        expected_combination_scan = Scan2D(
            pois,
            file_name_template_2d,
            input_subdirs,
            skip_best=True,
            best_fit_file=best_fit_file,
            model_config=models[args.model],
        )

    cat_names_string = "_".join(list(scan_dict.keys()))
    output_name = f"{args.model}_{cat_names_string}"
    if args.expected_bkg:
        output_name += "_expected_bkg"
    combination_name = combination
    if args.expected_bkg or args.expected:
        combination_name += "_asimov"
    if args.statonly:
        combination_name += "_statonly"
    legend_conf = {
        "loc": "upper left", 
        "prop": {"size": 18}
    }
    if args.model == "top_coupdep_ctcg":
        legend_conf = {
            "loc": "upper right",
            "prop": {"size": 14},
        }
    plot = TwoDScansPerModel(
        scan_dict=scan_dict,
        combination_name=combination_name,
        model_config=models[args.model],
        combination_asimov_scan=expected_combination_scan,
        output_name=output_name,
        is_asimov=args.expected,
        legend_conf=legend_conf
    )
    plot.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
