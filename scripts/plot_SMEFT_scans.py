import argparse
import numpy as np
import os
from itertools import combinations

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
from differential_combination_postprocess.figures import (
    GenericNLLsPerPOI,
    TwoDScansPerModel,
    SMEFTSummaryPlot
)
from differential_combination_postprocess.matrix import MatricesExtractor


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot SMEFT scans")

    parser.add_argument(
        "--how",
        required=True,
        choices=["freezeothers", "submodel"],
        help="Plot 1D scans for freezeothers or 1D + 2D scans for submodel",
    )

    parser.add_argument("--model", required=True, help="")

    parser.add_argument("--submodel", help="Path to yaml file indicating the submodel")

    parser.add_argument("--input-dir", type=str, default="inputs", help="")

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
        help="Categories for which the plots are produced",
    )

    parser.add_argument(
        "--coefficients",
        nargs="+",
        type=str,
        required=False,
        default=None,
        help="Wilson coefficients for which the plots are produced",
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

    parser.add_argument("--skip-2d", action="store_true", help="Skip 2D scans")

    parser.add_argument(
        "--config-file",
        type=str,
        required=False,
        default=None,
        help="Path to the configuration file containing cuts and stuff per POI per category",
    )

    parser.add_argument(
        "--force-2D-limit",
        action="store_true",
        help="Force the x and y limit on 2D scan to be the one from the submodel config file instead of the maximum between the scan and the submodel",
        default=False,
    )

    parser.add_argument(
        "--summary-plot",
        action="store_true",
        help="Produce a summary plot with all the 1D scans",
        default=False,
    )

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def plot_original_points(poi, scan_name, scan, subcategory, output_dir):
    fig, ax = plt.subplots()
    ax.plot(scan.original_points[0], scan.original_points[1], "ko", label=scan_name)
    ax.set_xlabel(poi)
    ax.set_ylabel("-2$\Delta$lnL")
    ax.legend()
    fig.savefig(f"{output_dir}/original_points_{poi}_{scan_name}_{subcategory}.png")
    fig.savefig(f"{output_dir}/original_points_{poi}_{scan_name}_{subcategory}.pdf")
    plt.close(fig)


oned_extra_selections = {
    "220926Atlas_ChbScen_PtFullComb": {
        "chb": lambda pois_values_original: ~np.logical_and(pois_values_original > 0.03, pois_values_original < 0.05)
    },
    "230620PruneNoCPEVPtFullCombLinearised_230620PruneNoCPEVPtFullCombLinearised_PtFullComb": {
        "EV3": lambda pois_values_original: ~np.logical_and(pois_values_original > 0.5, pois_values_original < 0.8),
        #"EV7": lambda pois_values_original: ~np.logical_and(pois_values_original > -15, pois_values_original < 10),
        #"EV8": lambda pois_values_original: ~np.logical_and(pois_values_original > 24, pois_values_original < 35),
    },
    "230620PruneNoCP_PtFullComb": {
        "cbwim": lambda pois_values_original: ~np.logical_or(
            np.logical_and(pois_values_original > -3.5, pois_values_original < -2),
            np.logical_and(pois_values_original > 1.6, pois_values_original < 2.5)
        ),
        "cbwre": lambda pois_values_original: ~np.logical_or(
            np.logical_and(pois_values_original > -5, pois_values_original < -4),
            np.logical_and(pois_values_original > 4.8, pois_values_original < 6.5)
        ),
        "cehim": lambda pois_values_original: ~np.logical_and(pois_values_original > 4.7, pois_values_original < 5.5),
        "chbox": lambda pois_values_original: ~np.logical_and(pois_values_original > 0.3, pois_values_original < 0.7),
        "chbq": lambda pois_values_original: ~np.logical_or(
            np.logical_and(pois_values_original > -2, pois_values_original < -1),
            np.logical_and(pois_values_original > 10, pois_values_original < 12)
        ),
        "chd": lambda pois_values_original: ~np.logical_or(
            np.logical_and(pois_values_original > -0.3, pois_values_original < -0.2),
            np.logical_and(pois_values_original > -0.05, pois_values_original < 0.1)
        ),
        "chg": lambda pois_values_original: ~np.logical_and(pois_values_original > 0.0035, pois_values_original < 0.0065),
        "chl3": lambda pois_values_original: ~np.logical_and(pois_values_original > -0.35, pois_values_original < -0.18),
        "chq3": lambda pois_values_original: ~np.logical_and(pois_values_original > -2, pois_values_original < -1.3),
        "cht": lambda pois_values_original: ~np.logical_or(
            np.logical_and(pois_values_original > -80, pois_values_original < -60),
            np.logical_and(pois_values_original > 20, pois_values_original < 40)
        ),
        "chw": lambda pois_values_original: ~np.logical_and(pois_values_original > -0.007, pois_values_original < -0.003),
        "chwb": lambda pois_values_original: ~np.logical_and(pois_values_original > 0.003, pois_values_original < 0.006),
        "cll1": lambda pois_values_original: ~np.logical_and(pois_values_original > 0.1, pois_values_original < 0.5),
        "ctgre": lambda pois_values_original: ~np.logical_and(pois_values_original > -0.1, pois_values_original < -0.05),
        "cuhre": lambda pois_values_original: ~np.logical_and(pois_values_original > 6, pois_values_original < 15),
    }
}


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")
    logger.info(
        f"Will run in {args.how} mode for model {args.model} and categories {args.categories}"
    )
    input_dir = os.path.join(args.input_dir, args.model)
    output_dir = os.path.join(args.output_dir, args.model)

    subcat = "observed"
    subcat_suff = ""
    if args.expected:
        subcat = "expected"
        subcat_suff = "_asimov"
    combination = args.combination

    cfg = {}
    if args.config_file is not None:
        cfg = extract_from_yaml_file(args.config_file)
    logger.debug(f"Configuration file: {cfg}")

    if args.how == "freezeothers":
        output_dir = os.path.join(output_dir, "freezeothers")
        os.makedirs(output_dir, exist_ok=True)
        # loop over wilson coefficients
        if args.coefficients:
            wilson_coefficients = args.coefficients
        else:
            wilson_coefficients = [
                dr.split("_")[-1]
                for dr in os.listdir(input_dir)
                if dr.startswith("FreezeOthers")
            ]
        logger.info(
            f"Will plot 1D scans for the following Wilson coefficients: {wilson_coefficients}"
        )

        for coeff in wilson_coefficients:
            input_dir_l = os.path.join(input_dir, f"FreezeOthers_{coeff}")
            scans = {}
            for category in args.categories:
                input_dirs = [
                    os.path.join(input_dir_l, d)
                    for d in os.listdir(input_dir_l)
                    if d.startswith(f"{category}{subcat_suff}-")
                ]
                if len(input_dirs) == 0:
                    logger.warning(
                        f"No input directories found for {category}{subcat_suff}"
                    )
                    continue
                extra_selection = None
                if "{}_{}{}".format(args.model, category, subcat_suff) in oned_extra_selections:
                    if coeff in oned_extra_selections["{}_{}{}".format(args.model, category, subcat_suff)]:
                        extra_selection = oned_extra_selections["{}_{}{}".format(args.model, category, subcat_suff)][coeff]

                scans[category] = Scan(
                    coeff, input_dirs, skip_best=True, allow_extrapolation=False, extra_selections=extra_selection
                )
                if args.expected_bkg:
                    logger.info(
                        "We will now look for the expected of args.combination, since we run with --expected-bkg"
                    )
                    cat = f"{combination}_asimov"
                    input_subdirs = [
                        os.path.join(input_dir_l, d)
                        for d in os.listdir(input_dir_l)
                        if d.startswith(f"{cat}-")
                    ]
                    #extra_selection = None
                    #if "{}_{}_{}".format(args.model, submodel_name, cat) in oned_extra_selections:
                    #    if coeff in oned_extra_selections["{}_{}_{}".format(args.model, submodel_name, cat)]:
                    #        extra_selection = oned_extra_selections["{}_{}_{}".format(args.model, submodel_name, cat)][coeff]
                    scans[cat] = Scan(
                        coeff,
                        input_subdirs,
                        skip_best=True,
                        allow_extrapolation=True,
                    )

            if len(scans) > 0:
                fig = GenericNLLsPerPOI(
                    coeff, scans, subcat, simple=True, plot_string=False
                )
                fig.dump(output_dir)
                fig = GenericNLLsPerPOI(
                    coeff, scans, subcat, simple=True, full_range=True
                )
                fig.dump(output_dir)
                logger.debug("Dumping original points")
                for scan_name, scan in scans.items():
                    plot_original_points(coeff, scan_name, scan, subcat, output_dir)

    elif args.how == "submodel":
        submodel_name = args.submodel.split("/")[-1].split(".")[0].split("_")[-1]
        model_plus_submodel_name = args.submodel.split("/")[-1].split(".")[0]
        if submodel_name not in os.listdir(input_dir):
            raise ValueError(f"No submodel {submodel_name} found in {input_dir}")
        output_dir = os.path.join(output_dir, submodel_name)
        os.makedirs(output_dir, exist_ok=True)
        input_dir = os.path.join(input_dir, submodel_name)
        submodel_extracted = extract_from_yaml_file(args.submodel)
        submodel_config = {}
        for key in [s for s in list(submodel_extracted.keys()) if s != "scan_ranges"]:
            submodel_config[key] = (
                submodel_extracted[key]["min"],
                submodel_extracted[key]["max"],
            )
        if args.coefficients:
            submodel_pois = args.coefficients
        else:
            submodel_pois = [
                s for s in list(submodel_extracted.keys()) if s != "scan_ranges"
            ]
        logger.info(f"Submodel {submodel_name} has the following POIs: {submodel_pois}")

        # corr and cov matrices
        # matrix_extractor = MatricesExtractor(submodel_pois)
        # matrix_extractor.extract_from_roofitresult(f"{input_dir}/multidimfitAsimovBestFit.root", "fit_mdf")
        # matrix_extractor.dump(output_dir)

        # first, one figure per wilson coefficient per subcategory with all the decay channels
        wc_scans = {}
        for coeff in submodel_pois:
            scans = {}
            for category in args.categories:
                input_dirs = [
                    os.path.join(input_dir, d)
                    for d in os.listdir(input_dir)
                    if d.startswith(f"{category}{subcat_suff}-")
                ]
                if len(input_dirs) == 0:
                    logger.warning(
                        f"No input directories found for {category}{subcat_suff}"
                    )
                    continue
                extra_selection = None
                if "{}_{}_{}{}".format(args.model, submodel_name, category, subcat_suff) in oned_extra_selections:
                    if coeff in oned_extra_selections["{}_{}_{}{}".format(args.model, submodel_name, category, subcat_suff)]:
                        extra_selection = oned_extra_selections["{}_{}_{}{}".format(args.model, submodel_name, category, subcat_suff)][coeff]
                scans[category] = Scan(
                    coeff,
                    input_dirs,
                    skip_best=True,
                    file_name_tmpl=f"higgsCombine_SCAN_1D{coeff}.*.root",
                    extra_selections=extra_selection,
                    cut_strings=cfg[model_plus_submodel_name][
                        f"{category}{subcat_suff}"
                    ][coeff]["cut_strings"]
                    if model_plus_submodel_name in cfg
                    and f"{category}{subcat_suff}" in cfg[model_plus_submodel_name]
                    and coeff
                    in cfg[model_plus_submodel_name][f"{category}{subcat_suff}"]
                    else None,
                    allow_extrapolation=False,
                )
            if args.expected_bkg:
                logger.info(
                    "We will now look for the expected of args.combination, since we run with --expected-bkg"
                )
                cat = f"{combination}_asimov"
                input_subdirs = [
                    os.path.join(input_dir, d)
                    for d in os.listdir(input_dir)
                    if d.startswith(f"{cat}-")
                ]
                extra_selection = None
                if "{}_{}_{}".format(args.model, submodel_name, cat) in oned_extra_selections:
                    if coeff in oned_extra_selections["{}_{}_{}".format(args.model, submodel_name, cat)]:
                        extra_selection = oned_extra_selections["{}_{}_{}".format(args.model, submodel_name, cat)][coeff]
                scans[cat] = Scan(
                    coeff,
                    input_subdirs,
                    skip_best=True,
                    file_name_tmpl=f"higgsCombine_SCAN_1D{coeff}.*.root",
                    extra_selections=extra_selection,
                    cut_strings = cfg[model_plus_submodel_name][
                        f"{cat}"
                    ][coeff]["cut_strings"] if model_plus_submodel_name in cfg and f"{cat}" in cfg[model_plus_submodel_name] and coeff in cfg[model_plus_submodel_name][f"{cat}"] else None,
                    allow_extrapolation=False,
                )
            if len(scans) > 0:
                fig = GenericNLLsPerPOI(
                    coeff, scans, subcat, simple=True, plot_string=True
                )
                fig.dump(output_dir)
                fig = GenericNLLsPerPOI(
                    coeff, scans, subcat, simple=True, full_range=True
                )
                fig.dump(output_dir)
                logger.debug("Dumping original points")
                for scan_name, scan in scans.items():
                    plot_original_points(coeff, scan_name, scan, subcat, output_dir)
            wc_scans[coeff] = scans
        if args.summary_plot:
            logger.info("Will now produce the summary plot")
            summary_plot = SMEFTSummaryPlot(
                wc_scans,
            )
            summary_plot.dump(output_dir)

        # then, 2D plots
        if not args.skip_2d and args.coefficients is None:
            pairs = list(combinations(submodel_pois, 2))
            logger.info(f"Will plot 2D scans for the following pairs of WCs: {pairs}")
            for pair in pairs:
                # expected with gradient and observed with lines for each category
                scans = {}
                for category in args.categories:
                    input_dirs = [
                        os.path.join(input_dir, d)
                        for d in os.listdir(input_dir)
                        if d.startswith(f"{category}{subcat_suff}-")
                    ]
                    if len(input_dirs) == 0:
                        logger.warning(
                            f"No input directories found for {category}{subcat_suff}"
                        )
                        continue
                    try:
                        scans[f"{category}{subcat_suff}"] = Scan2D(
                            pois=pair,
                            file_name_template=f"higgsCombine_SCAN_2D{pair[0]}-{pair[1]}.*.root",
                            input_dirs=input_dirs,
                            skip_best=True,
                            best_fit_file=f"{input_dirs[0]}/higgsCombineAsimovBestFit.MultiDimFit.mH125.38.root"
                            if "asimov" in subcat_suff
                            else None,
                        )
                    except Exception as e:
                        pass
                        logger.warning(
                            f"Could not load 2D scan for {pair} in {category}{subcat_suff}"
                        )
                        logger.warning(e)
                if len(scans) > 0:
                    logger.debug(f"Found scan dictionary {scans}")
                    pair_submodel_config = {}
                    for par in pair:
                        pair_submodel_config[par] = submodel_config[par]
                    # if we want the expected as bkg, make the scan
                    expected_combination_scan = None
                    if args.expected_bkg:
                        logger.info(
                            "We will now look for the expected of args.combination, since we run with --expected-bkg"
                        )
                        cat = f"{combination}_asimov"
                        input_subdirs = [
                            os.path.join(input_dir, d)
                            for d in os.listdir(input_dir)
                            if d.startswith(f"{cat}-")
                        ]
                        best_fit_file = os.path.join(
                            input_subdirs[0],
                            "higgsCombineAsimovBestFit.MultiDimFit.mH125.38.root",
                        )
                        expected_combination_scan = Scan2D(
                            pair,
                            file_name_template=f"higgsCombine_SCAN_2D{pair[0]}-{pair[1]}.*.root",
                            input_dirs=input_subdirs,
                            skip_best=True,
                            best_fit_file=best_fit_file,
                        )
                        expected_combination_scan.category = combination

                    fig = TwoDScansPerModel(
                        scan_dict=scans,
                        combination_name=f"{combination}_asimov"
                        if "asimov" in subcat_suff
                        else combination,
                        model_config=pair_submodel_config,
                        combination_asimov_scan=expected_combination_scan,
                        output_name=f"2D_{pair[0]}-{pair[1]}_{'_'.join(args.categories)}{subcat_suff}{'_expected_bkg' if args.expected_bkg else ''}",
                        is_asimov=True if "asimov" in subcat_suff else False,
                        force_limit=args.force_2D_limit,
                    )
                    fig.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
