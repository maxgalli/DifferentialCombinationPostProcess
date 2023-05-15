import argparse

# Needed to fix the fucking
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("AGG")

from differential_combination_postprocess.utils import (
    setup_logging,
    extract_from_yaml_file,
)
from differential_combination_postprocess.matrix import MatricesExtractor


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--rfr-file", type=str, default="", help="")
    parser.add_argument("--robusthesse-file", type=str, default="", help="")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where output files will be stored",
    )

    parser.add_argument("--pois", nargs="+", type=str, required=True, help="")

    parser.add_argument("--suffix", type=str, default="", help="")

    parser.add_argument(
        "--observable",
        type=str,
        default=None,
        help="Observable, mostly used for signal strength measurements; plot the observable name on the y axis",
    )

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    if len(args.pois) == 1 and args.pois[0].endswith(".yml"):
        pois = extract_from_yaml_file(args.pois[0])
    else:
        pois = args.pois
    logger.debug("POIs: {}".format(pois))
    me = MatricesExtractor(pois)
    if args.rfr_file:
        me.extract_from_roofitresult(args.rfr_file, "fit_mdf")
    if args.robusthesse_file:
        me.extract_from_robusthesse(args.robusthesse_file)
    me.dump(args.output_dir, args.suffix, observable=args.observable)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
