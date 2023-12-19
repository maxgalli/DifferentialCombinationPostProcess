import argparse
import math
import numpy as np

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
from differential_combination_postprocess.cosmetics import get_parameter_label


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

    parser.add_argument(
        "--file-name-tmpl",
        type=str,
        required=False,
        default=None,
        help="File name templates",
    )

    parser.add_argument(
        "--other-file-name-tmpl",
        type=str,
        required=False,
        default=None,
        help="File name templates",
    )

    parser.add_argument("--cut-strings", type=str, nargs="+", default=None, help="Cut strings")

    parser.add_argument("--other-cut-strings", type=str, nargs="+", default=None, help="Cut strings")
    
    parser.add_argument("--output-dir", type=str, help="")

    parser.add_argument("--poi", type=str, help="POI to plot in single scan")

    parser.add_argument("--stat-syst", action="store_true", help="Plot stat+syst instead of syst-only")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def frexp10(x):
    exp = int(math.floor(math.log10(abs(x))))
    return x / 10 ** exp, exp

    
def get_result_label(scan_full, scan_statonly):
    minimum = scan_full.minimum[0]
    minimum_statonly = scan_statonly.minimum[0]
    up = scan_full.up68_unc[0]
    #print(up)
    down = scan_full.down68_unc[0]
    up_statonly = scan_statonly.up68_unc[0]
    #print(up_statonly)
    down_statonly = scan_statonly.down68_unc[0]
    up_syst = np.abs(minimum * np.sqrt((up / minimum) ** 2 - (up_statonly / minimum) ** 2))
    #print(up_syst)
    down_syst = np.abs(minimum * np.sqrt((down / minimum) ** 2 - (down_statonly / minimum) ** 2))

    _, exp = frexp10(up)
    if exp < 0:
        up /= 10 ** exp
        down /= 10 ** exp
        up_statonly /= 10 ** exp
        down_statonly /= 10 ** exp
        up_syst /= 10 ** exp
        down_syst /= 10 ** exp
    
        # make string
        poi_label = get_parameter_label(scan_full.poi)
        upstring_syst = "{+" + f"{up_syst:.2f}" + "x10^{" + f"{exp}" + "}" + "}"
        downstring_syst = "{-" + f"{down_syst:.2f}" + "x10^{" + f"{exp}" + "}" + "}"
        upstring_statonly = "{+" + f"{up_statonly:.2f}" + "x10^{" + f"{exp}" + "}" + "}"
        downstring_statonly = "{-" + f"{down_statonly:.2f}" + "x10^{" + f"{exp}" + "}" + "}"
        upstring = "{+" + f"{up:.2f}" + "x10^{" + f"{exp}" + "}" + "}"
        downstring = "{-" + f"{down:.2f}" + "x10^{" + f"{exp}" + "}" + "}"
        result_label = f"{poi_label} = ${minimum:.2f}^{upstring_syst}_{downstring_syst} (syst)^{upstring_statonly}_{downstring_statonly} (stat)^{upstring_statonly}_{downstring_statonly}"
        result_label += f" = {minimum:.2f}^{upstring}_{downstring} (syst+stat)$"
    
    else:
        # make string
        poi_label = get_parameter_label(scan_full.poi)
        upstring_syst = "{+" + f"{up_syst:.2f}" + "}"
        downstring_syst = "{-" + f"{down_syst:.2f}" + "}"
        upstring_statonly = "{+" + f"{up_statonly:.2f}" + "}"
        downstring_statonly = "{-" + f"{down_statonly:.2f}" + "}"
        upstring = "{+" + f"{up:.2f}" + "}"
        downstring = "{-" + f"{down:.2f}" + "}"
        result_label = f"{poi_label} = ${minimum:.2f}^{upstring_syst}_{downstring_syst} (syst)^{upstring_statonly}_{downstring_statonly} (stat)"
        result_label += f" = {minimum:.2f}^{upstring}_{downstring} (syst+stat)$"
    
    return result_label
    


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    scans = {}
    for label, input_dir, tmpl, cs in zip(
        [args.label, args.other_label],
        [args.input_dir, args.other_input_dir],
        [args.file_name_tmpl, args.other_file_name_tmpl],
        [args.cut_strings, args.other_cut_strings],
    ):
        scans[label] = Scan(
            poi=args.poi,
            input_dirs=[input_dir],
            skip_best=True,
            file_name_tmpl=tmpl,
            cut_strings=cs,
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
        bestfit1=False if args.stat_syst else True,
        bestfit2=False if args.stat_syst else True,
    )
    if args.stat_syst:
        result_label = get_result_label(scans[args.label], scans[args.other_label])
        # add it to the figure
        fig.ax.text(
        0.5,
        0.75,
        result_label,
        color="k",
        fontsize=12,
        ha="center",
        va="center",
        transform=fig.ax.transAxes,
    )
    
    fig.dump(args.output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
