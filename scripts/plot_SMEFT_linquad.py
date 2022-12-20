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

    parser.add_argument(
        "--linear", help="Path to yaml file indicating the linear submodel"
    )

    parser.add_argument(
        "--quad", help="Path to yaml file indicating the quadratic submodel"
    )

    parser.add_argument("--input-dir", type=str, default="inputs", help="")

    parser.add_argument("--category", type=str, help="E.g. Hgg, Hgg_asimov, etc.")

    parser.add_argument("--observable", type=str, help="")

    parser.add_argument("--output-dir", type=str, help="")

    parser.add_argument(
        "--config-file",
        type=str,
        required=False,
        default=None,
        help="Path to the configuration file containing cuts and stuff per POI per category",
    )

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    output_dir_name = ""

    submodels = {"linear": args.linear, "quadratic": args.quad}

    category = args.category

    scans = {}

    cfg = {}
    if args.config_file is not None:
        cfg = extract_from_yaml_file(args.config_file)
    logger.debug(f"Configuration file: {cfg}")

    for how, submodel_path in submodels.items():
        submodel_name = submodel_path.split("/")[-1].split(".")[0].split("_")[-1]
        model_name = submodel_path.split("/")[-1].split(".")[0].split("_")[0]
        model_plus_submodel_name = (
            f"{model_name}_{submodel_name}"
            if submodel_name != model_name
            else model_name
        )
        output_dir_name += f"{model_plus_submodel_name}_"

        input_dir = os.path.join(
            args.input_dir, args.observable, model_name, submodel_name
        )
        submodel_extracted = extract_from_yaml_file(submodel_path)

        scans[how] = {}

        for coeff in submodel_extracted:
            input_dirs = [
                os.path.join(input_dir, d)
                for d in os.listdir(input_dir)
                if d.startswith(f"{category}-")
            ]
            if len(input_dirs) == 0:
                logger.warning(f"No input directories found for {category}")
                continue
            scans[how][coeff] = Scan(
                coeff,
                input_dirs,
                skip_best=True,
                file_name_tmpl=f"higgsCombine_SCAN_1D{coeff}.*.root",
                cut_strings=cfg[model_plus_submodel_name][category][coeff][
                    "cut_strings"
                ]
                if model_plus_submodel_name in cfg
                and category in cfg[model_plus_submodel_name]
                and coeff in cfg[model_plus_submodel_name][category]
                else None,
                allow_extrapolation=False,
            )

    output_dir = os.path.join(args.output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    for coeff in scans["linear"]:
        if coeff in scans["quadratic"]:
            logger.info(f"Plotting {coeff}")
            fig = TwoScans(
                coeff,
                category,
                scans["linear"][coeff],
                scans["quadratic"][coeff],
                "Linear",
                "Quadratic",
                "LinVsQuad",
                bestfit1=True,
                bestfit2=False,
            )
            fig.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
