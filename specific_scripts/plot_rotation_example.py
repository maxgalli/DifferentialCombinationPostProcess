# Needed to fix the fucking
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

matplotlib.use("AGG")
hep.style.use("CMS")

from differential_combination_postprocess.scan import Scan, Scan2D
from differential_combination_postprocess.figures import (
    GenericNLLsPerPOI,
    TwoDScansPerModel,
)
from differential_combination_postprocess.utils import (
    setup_logging,
    extract_from_yaml_file,
)


def make_figure(scan, labels):
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax, colormap, pc = scan.plot_as_heatmap(ax)
    fig.colorbar(pc, ax=ax, label="-2$\Delta$lnL")
    ax.legend(loc="upper left")
    hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=ax)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    return fig, ax


def main():
    logger = setup_logging(level="INFO")
    input_dirs = [
        "outputs/SMEFT_scans/smH_PTH/230105MiniTwo/230105MiniTwo/Hgg_asimov-20230105xxx150343",
        "outputs/SMEFT_scans/smH_PTH/230105MiniTwoEVhgg/230105MiniTwoEVhgg/Hgg_asimov-20230105xxx145310",
    ]
    pairs = [["chg", "chb"], ["EV0", "EV1"]]

    output_dir = "/eos/home-g/gallim/www/plots/DifferentialCombination/CombinationRun2/AN_plots/other"

    category = "Hgg_asimov"

    model_configs = [
        {"chg": (-0.1, 0.1), "chb": (-0.05, 0.05)},
        {"EV0": (-0.005, 0.005), "EV1": (-0.1, 0.1)},
    ]

    labels = [
        ["$c_{HG}$", "$c_{HB}$"],
        ["$0.61 c_{HG} - 0.80 c_{HB}$", "$-0.80 c_{HG} - 0.61c_{HG}$"],
    ]

    for input_dir, pair, mc, lbs in zip(input_dirs, pairs, model_configs, labels):
        scan = Scan2D(
            pois=pair,
            file_name_template=f"higgsCombine_SCAN_2D{pair[0]}-{pair[1]}.*.root",
            input_dirs=[input_dir],
            skip_best=True,
            best_fit_file=f"{input_dir}/higgsCombineAsimovBestFit.MultiDimFit.mH125.38.root",
            model_config=mc,
        )

        fig, ax = make_figure(scan, lbs)
        fig.savefig(
            f"{output_dir}/example_{category}_{pair[0]}-{pair[1]}.png",
            bbox_inches="tight",
        )
        fig.savefig(
            f"{output_dir}/example_{category}_{pair[0]}-{pair[1]}.pdf",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
