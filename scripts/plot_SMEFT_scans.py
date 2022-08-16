import argparse
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
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot SMEFT scans")

    parser.add_argument(
        "--observable",
        default="smH_PTH",
        type=str,
        help="observable to produce XS plots for",
    )

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
        required=False,
        default=[],
        help="Categories for which the plots are produced",
    )

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")
    logger.info(
        f"Will run in {args.how} mode for model {args.model}, categories {args.categories} (and observable {args.observable})"
    )
    input_dir = os.path.join(args.input_dir, args.observable, args.model)
    output_dir = os.path.join(args.output_dir, args.observable, args.model)

    subcategory_suff = {"observed": "", "expected": "_asimov"}

    if args.how == "freezeothers":
        output_dir = os.path.join(output_dir, "freezeothers")
        os.makedirs(output_dir, exist_ok=True)
        # loop over wilson coefficients
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
            for subcat, subcat_suff in subcategory_suff.items():
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
                    scans[category] = Scan(coeff, input_dirs, skip_best=True)
                if len(scans) > 0:
                    fig = GenericNLLsPerPOI(coeff, scans, subcat)
                    fig.dump(output_dir)

    elif args.how == "submodel":
        submodel_name = args.submodel.split("/")[-1].split(".")[0].split("_")[-1]
        if submodel_name not in os.listdir(input_dir):
            raise ValueError(f"No submodel {submodel_name} found in {input_dir}")
        output_dir = os.path.join(output_dir, submodel_name)
        os.makedirs(output_dir, exist_ok=True)
        input_dir = os.path.join(input_dir, submodel_name)
        submodel_extracted = extract_from_yaml_file(args.submodel)
        submodel_config = {}
        for key in submodel_extracted:
            submodel_config[key] = (
                submodel_extracted[key]["min"],
                submodel_extracted[key]["max"],
            )
        submodel_pois = list(submodel_extracted.keys())
        logger.info(f"Submodel {submodel_name} has the following POIs: {submodel_pois}")

        # first, one figure per wilson coefficient per subcategory with all the decay channels
        for coeff in submodel_pois:
            for subcat, subcat_suff in subcategory_suff.items():
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
                    scans[category] = Scan(
                        coeff,
                        input_dirs,
                        skip_best=False,
                        file_name_tmpl=f"higgsCombine_SCAN_1D{coeff}*.root",
                    )
                if len(scans) > 0:
                    fig = GenericNLLsPerPOI(coeff, scans, subcat)
                    fig.dump(output_dir)

        # then, 2D plots
        pairs = list(combinations(submodel_pois, 2))
        logger.info(f"Will plot 2D scans for the following pairs of WCs: {pairs}")
        for pair in pairs:
            # expected with gradient and observed with lines for each category
            for category in args.categories:
                scans = {}
                for subcat, subcat_suff in subcategory_suff.items():
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
                    scans[f"{category}{subcat_suff}"] = Scan2D(
                        pois=pair,
                        file_name_template=f"higgsCombine_SCAN_2D{pair[0]}-{pair[1]}*.root",
                        input_dirs=input_dirs,
                        skip_best=True,
                    )
                if len(scans) > 0:
                    logger.debug(f"Found scan dictionary {scans}")
                    fig = TwoDScansPerModel(
                        scan_dict=scans,
                        combination_name=f"{category}_asimov",
                        model_config=submodel_config,
                        output_name=f"2D_{pair[0]}-{pair[1]}_{category}",
                    )
                    fig.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
