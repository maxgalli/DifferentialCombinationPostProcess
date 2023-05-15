# Needed to fix the fucking
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.interpolate import griddata
import pickle
from differential_combination_postprocess.utils import setup_logging

matplotlib.use("AGG")
hep.style.use("CMS")


def plot(result, pois, labels):
    poi1, poi2 = pois
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    x, y = np.mgrid[
        result[f"{poi1}_{poi2}"]["values1"]
        .min() : result[f"{poi1}_{poi2}"]["values1"]
        .max() : 100j,
        result[f"{poi1}_{poi2}"]["values2"]
        .min() : result[f"{poi1}_{poi2}"]["values2"]
        .max() : 100j,
    ]
    z = griddata(
        (result[f"{poi1}_{poi2}"]["values1"], result[f"{poi1}_{poi2}"]["values2"]),
        result[f"{poi1}_{poi2}"]["chi_square"],
        (x, y),
        method="cubic",
        fill_value=10.0,
    )
    colormap = plt.get_cmap("plasma")
    # colormap = colormap.reversed()
    pc = ax.pcolormesh(x, y, z, vmin=0, cmap=colormap, shading="gouraud")
    fig.colorbar(pc, ax=ax, label="$\Delta\chi^2$", fraction=0.047, pad=0.01)

    ax.legend(loc="upper left")
    hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=ax)
    fig.tight_layout()
    ax.set_xlabel(labels[0], fontsize=28)
    ax.set_ylabel(labels[1], fontsize=28)

    return fig, ax


def main():
    logger = setup_logging(level="INFO")
    input_files = [
        "/work/gallim/DifferentialCombination_home/EFTModelsStudies/outputs/results_230105MiniTwo_hgg_linearised.pkl",
        "/work/gallim/DifferentialCombination_home/EFTModelsStudies/outputs/results_230105MiniTwoEVhgg_hgg_linearised.pkl",
    ]
    pairs = [["chg", "chb"], ["EV0", "EV1"]]
    output_dir = "/eos/home-g/gallim/www/plots/DifferentialCombination/CombinationRun2/AN_plots/other"
    labels = [
        ["$c_{HG}$", "$c_{HB}$"],
        ["$0.61 c_{HG} - 0.80 c_{HB}$", "$-0.80 c_{HG} - 0.61c_{HG}$"],
    ]
    for input_dir, pair, lbs in zip(input_files, pairs, labels):
        with open(input_dir, "rb") as f:
            results = pickle.load(f)
        result = results["expected_fixed"]
        fig, ax = plot(result, pair, lbs)
        for format in ["png", "pdf"]:
            fig.savefig(f"{output_dir}/example_chi_{pair[0]}_{pair[1]}.{format}")


if __name__ == "__main__":
    main()
