""" Script to plot NLLs and final (superimposed) plots for a single observable.
Following are the main parameters:
- 
"""
import argparse
import os
import re
import numpy as np
from copy import deepcopy
import pickle as pkl

# Needed to fix the fucking
# _tkinter.TclError: couldn't connect to display "localhost:12.0"
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("AGG")

from differential_combination_postprocess.utils import (
    setup_logging,
    extract_from_yaml_file,
    TK_parser,
)
from differential_combination_postprocess.scan import Scan, DifferentialSpectrum
from differential_combination_postprocess.figures import (
    XSNLLsPerCategory,
    XSNLLsPerPOI,
    XSNLLsPerPOI_Full,
    DiffXSsPerObservable,
)
from differential_combination_postprocess.shapes import (
    ObservableShapeFitted,
    sm_shapes,
    sm_shapes_noNNLOPS,
    sm_shapes_powheg,
    smH_PTH_EvenMoreMaximumGranularity_obs_shape,
    yH_Granular_obs_shape,
    ObservableShapeKappa,
)
from differential_combination_postprocess.physics import analyses_edges, overflows

import logging


def parse_arguments():
    parser = argparse.ArgumentParser(description="Produce XS plots")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    parser.add_argument(
        "--observable",
        required=True,
        type=str,
        help="observable to produce XS plots for",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="inputs",
        help="Directory where the .root files with 'limit' trees are stored",
    )

    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="outputs",
        help="Directory where the .yaml files with metadata are stored",
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
        help="Categories for which NLL plots are dumped, along with the superimposed final ones",
    )

    parser.add_argument(
        "--singles",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="Categories for which fits were performed with Combines' singles",
    )

    parser.add_argument(
        "--systematic-bands",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="Categories for which we want to plot the systematics bands in the final plot",
    )

    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        type=str,
        default=[],
        help="Directory to exclude from the list of input directories. Note that since I'm retarded it only needs the name of the folder (no previous path), otherwise it breaks",
    )

    parser.add_argument(
        "--no-final",
        action="store_true",
        default=False,
        help="Do not produce final plots",
    )

    parser.add_argument(
        "--no-nll",
        action="store_true",
        default=False,
        help="Skip production of NLL plots",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        required=False,
        default=None,
        help="Path to the configuration file containing cuts and stuff per POI per category",
    )

    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        default=False,
        help="Whether or not extrapolating if NLL does not cross 1",
    )

    parser.add_argument(
        "--align-statonly",
        action="store_true",
        default=False,
        help="Whether or not aligning the statonly scans to the nominal one",
    )

    parser.add_argument(
        "--kappa-prediction",
        type=str,
        required=False,
        help="Path to a TK-style kappa prediction file",
    )

    return parser.parse_args()


def get_shapes_from_differential_spectra(differential_spectra, observable):
    shapes = []
    for category, spectrum in differential_spectra.items():
        simple_category = category.split("_")[0]
        # Remember that the results we get for mu are meant to scale the SM cross section
        logging.info(
            f"Building shape for category {category}, observable {observable} and edges {analyses_edges[observable][simple_category]}"
        )
        # First: copy the finest possible shape (Hgg) and rebin it with what we need
        sm_rebinned_shape = deepcopy(sm_shapes[observable])
        # Fuckin Hbb porcaccioddio
        if simple_category == "Hbb" and observable == "smH_PTH":
            sm_rebinned_shape = deepcopy(smH_PTH_EvenMoreMaximumGranularity_obs_shape)
        logging.debug(f"SM original xs: {sm_rebinned_shape.xs}")
        logging.debug(
            f"SM original xs_over_binwidth: {sm_rebinned_shape.xs_over_bin_width}"
        )
        sm_rebinned_shape.rebin(analyses_edges[observable][simple_category])
        # Second: bin-wise multiply the new bin values by the values for mu computed in the scan
        if spectrum.from_singles:
            mus = np.array([scan.mu for scan in spectrum.scans.values()])
            mus_up = np.array([scan.mu_up for scan in spectrum.scans.values()])
            mus_down = np.array([scan.mu_down for scan in spectrum.scans.values()])
        else:
            mus = np.array([scan.minimum[0] for scan in spectrum.scans.values()])
            mus_up = np.array(
                [scan.minimum[0] + scan.up68_unc[0] for scan in spectrum.scans.values()]
            )
            mus_down = np.array(
                [
                    scan.minimum[0] - scan.down68_unc[0]
                    for scan in spectrum.scans.values()
                ]
            )
        logging.debug(f"Ordered mus for category {category}: {mus}")
        logging.debug(f"Ordered mus up for category {category}: {mus_up}")
        logging.debug(f"Ordered mus down for category {category}: {mus_down}")
        logging.debug(f"SM rebinned xs: {sm_rebinned_shape.xs}")
        logging.debug(
            f"SM rebinned xs_over_binwidth: {sm_rebinned_shape.xs_over_bin_width}"
        )
        weighted_bins = np.multiply(np.array(sm_rebinned_shape.xs), mus)
        logging.debug(f"Reweighted xs: {weighted_bins}")
        weighted_bins_up = np.multiply(np.array(sm_rebinned_shape.xs), mus_up)
        weighted_bins_down = np.multiply(np.array(sm_rebinned_shape.xs), mus_down)
        logging.debug(f"Reweighted xs up: {weighted_bins_up}")
        logging.debug(f"Reweighted xs down: {weighted_bins_down}")
        shapes.append(
            ObservableShapeFitted(
                observable,
                simple_category,
                analyses_edges[observable][simple_category],
                weighted_bins,
                weighted_bins_up,
                weighted_bins_down,
                overflow=observable in overflows,
            )
        )
        logging.debug(f"Just appended shape {shapes[-1]}")

    return shapes

oned_extra_selection = {
    "smH_PTH_FinalComb_statonly": {
        "r_smH_PTH_20_25": lambda pois_values_original: ~np.logical_and(pois_values_original > 0.3, pois_values_original < 0.55)
    }
}


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    observable = args.observable
    input_dir = args.input_dir
    metadata_dir = args.metadata_dir
    output_dir = args.output_dir
    categories = args.categories
    if len(categories) == 1 and categories[0] == "FinalComb":
        logger.info("Updating SM shapes since the only category is FinalComb")
        from differential_combination_postprocess.shapes import smH_PTH_Hgg_obs_shape, smH_PTH_Hgg_obs_shape_noNNLOPS, smH_PTH_Hgg_obs_shape_powheg
        sm_shapes["smH_PTH"] = smH_PTH_Hgg_obs_shape
        sm_shapes_noNNLOPS["smH_PTH"] = smH_PTH_Hgg_obs_shape_noNNLOPS
        sm_shapes_powheg["smH_PTH"] = smH_PTH_Hgg_obs_shape_powheg

        from differential_combination_postprocess.shapes import smH_PTJ0_Hgg_obs_shape, smH_PTJ0_Hgg_obs_shape_noNNLOPS, smH_PTJ0_Hgg_obs_shape_powheg
        sm_shapes["smH_PTJ0"] = smH_PTJ0_Hgg_obs_shape
        sm_shapes_noNNLOPS["smH_PTJ0"] = smH_PTJ0_Hgg_obs_shape_noNNLOPS
        sm_shapes_powheg["smH_PTJ0"] = smH_PTJ0_Hgg_obs_shape_powheg
         
    singles = args.singles
    if len(categories + singles) == 0:
        raise ValueError("Please specify at least one category or singles")
    systematic_bands = args.systematic_bands
    exclude_dirs = args.exclude_dirs
    cfg = {}
    if args.config_file is not None:
        cfg = extract_from_yaml_file(args.config_file)

    logger.info(f"Plotting session for observable {observable}")

    # First produce NLL plots, one for each category
    # Each containing the NLL curves for each POI
    logger.info(f"Working with the following categories: {categories}")
    categories_yamls = ["{}.yml".format(category) for category in categories + singles]
    logger.debug(f"YAMLs: {categories_yamls}")

    differential_spectra = {}
    differential_spectra_statonly = {}
    differential_spectra_asimov = {}
    differential_spectra_asimov_statonly = {}

    for fl in categories_yamls:
        full_path_to_file = f"{metadata_dir}/{observable}/{fl}"
        logger.debug(f"Full path to file: {full_path_to_file}")
        # Based on the assumption that we have a config file for each category called 'category_name.yaml'
        category = fl.split(".")[0]
        pois = extract_from_yaml_file(full_path_to_file)
        # in mjj we fit r_out that we don't plot
        if observable in ["mjj", "DEtajj"]:
            pois = pois[1:]
        elif observable == "TauCJ" and category in ["Hgg", "HZZ"]:
            pois = pois[1:]
        elif observable == "TauCJ" and category in ["HggHZZ"]:
            pois = pois[2:]

        # Here define categories for asimov, statonly and asimov_statonly
        asimov_cat = f"{category}_asimov"
        statonly_cat = f"{category}_statonly"
        asimov_statonly_cat = f"{category}_asimov_statonly"

        sub_categories = [category, asimov_cat, statonly_cat, asimov_statonly_cat]
        if args.no_nll:
            sub_categories = [category, statonly_cat]

        # Plot scans for nominal, statonly, asimov and asimov_statonly
        sub_cat_spectra = {}
        for sub_cat in sub_categories:
            logger.info(f"Working on sub-category {sub_cat}")
            categories_numbers = [
                directory
                for directory in os.listdir(input_dir)
                if directory.startswith(f"{sub_cat}-")
            ]
            # Give the possibility to exclude some directories like the ones for which jobs are still running
            categories_numbers = [
                directory
                for directory in categories_numbers
                if directory not in exclude_dirs
            ]
            if len(categories_numbers) == 0:
                logger.warning(f"No directories found for category {sub_cat}")
                continue
            logger.debug(
                f"Will look into the following directories: {categories_numbers}"
            )
            category_input_dirs = [
                f"{input_dir}/{directory}" for directory in categories_numbers
            ]

            spectrum_align_to = None
            pois_align_to = None
            if args.align_statonly and "statonly" in sub_cat:
                if "asimov" in sub_cat:
                    align_to = asimov_cat
                else:
                    align_to = category
                spectrum_align_to = sub_cat_spectra[align_to]
                pois_align_to = pois
            if sub_cat in cfg:
                if "align_to" in cfg[sub_cat] and "statonly" in sub_cat:
                    if "asimov" in sub_cat:
                        align_to = asimov_cat
                    else:
                        align_to = category
                    spectrum_align_to = sub_cat_spectra[align_to]
                    pois_align_to = cfg[sub_cat]["align_to"]

            extra_selections = None
            key = f"{observable}_{sub_cat}"
            if key in oned_extra_selection:
                extra_selections = oned_extra_selection[key]

            diff_spectrum = DifferentialSpectrum(
                observable,
                sub_cat,
                pois,
                category_input_dirs,
                from_singles=category in singles,
                skip_best=cfg[sub_cat]["skip_best"]
                if (sub_cat in cfg and "skip_best" in cfg[sub_cat])
                else False,
                cut_strings={
                    p: cfg[sub_cat][p]["cut_strings"] for p in pois if p in cfg[sub_cat]
                }
                if sub_cat in cfg
                else None,
                allow_extrapolation=args.allow_extrapolation,
                align_to=(spectrum_align_to, pois_align_to),
                extra_selections=extra_selections,
            )

            sub_cat_spectra[sub_cat] = diff_spectrum
            if sub_cat == category:
                differential_spectra[sub_cat] = diff_spectrum
            if sub_cat == statonly_cat and sub_cat.split("_")[0] in systematic_bands:
                differential_spectra_statonly[sub_cat] = diff_spectrum
            if sub_cat == asimov_cat:
                differential_spectra_asimov[sub_cat] = diff_spectrum
            if (
                sub_cat == asimov_statonly_cat
                and sub_cat.split("_")[0] in systematic_bands
            ):
                differential_spectra_asimov_statonly[sub_cat] = diff_spectrum

            if not args.no_nll and not diff_spectrum.from_singles:
                plot_to_dump = XSNLLsPerCategory(diff_spectrum)
                plot_to_dump.dump(output_dir)
                plt.close("all")

        # Plot one figure per POI with nominal, statonly, asimov and asimov_statonly NLLs
        if not args.no_nll and not diff_spectrum.from_singles:
            logger.info(
                "Now plotting one figure per POI with nominal, statonly, asimov and asimov_statonly NLLs (if found)"
            )
            poi_plots = XSNLLsPerPOI(sub_cat_spectra, debug=args.debug)
            poi_plots.dump(output_dir)

        if args.debug and not args.no_nll and not diff_spectrum.from_singles:
            full_plot_to_dump = XSNLLsPerPOI_Full(sub_cat_spectra)
            full_plot_to_dump.dump(output_dir)

    ds_full_list = [differential_spectra, differential_spectra_asimov]
    ds_statonly_list = [
        differential_spectra_statonly,
        differential_spectra_asimov_statonly,
    ]

    if "inclusive" not in (categories + singles)[0]:
        for i, (ds_full, ds_statonly) in enumerate(zip(ds_full_list, ds_statonly_list)):
            logger.debug(f"Differential spectra: {ds_full}")

            # Produce the final differential xs plot including all the categories
            logger.info(
                f"Now producing the final differential xs plot for observable {observable}"
            )

            shapes = get_shapes_from_differential_spectra(ds_full, observable)
            shapes_statonly = get_shapes_from_differential_spectra(
                ds_statonly, observable
            )
            # horrible
            # I should probably introduce another dict
            shapes_systonly = []
            for shape in shapes:
                for shape_statonly in shapes_statonly:
                    if (
                        shape.category == shape_statonly.category
                        and shape.observable == shape_statonly.observable
                    ):
                        #shapes_systonly.append(np.sqrt(shape**2 - shape_statonly**2))
                        shape_syst = shape - shape_statonly
                        shapes_systonly.append(shape_syst)
            if len(shapes) < 2:
                for name, shapes_list in zip(["", "_statonly", "_systonly"], [shapes, shapes_statonly, shapes_systonly]):
                    for shape in shapes_list:
                        to_dump = {
                            "pois": pois,
                            "shape": shape
                        }
                        output_name = f"{output_dir}/{observable}_{shape.category}{'_asimov' if i == 1 else ''}{name}.pkl"
                        logger.info(f"Dumping shape for category {shape.category} to {output_name}")
                        with open(output_name, "wb") as f:
                            pkl.dump(to_dump, f)
                       
            if args.debug:
                for shape in shapes_systonly:
                    logger.debug(f"Systematic shape: \n{shape}")

            if not args.no_final:
                other_sm_shapes_dicts = None
                if len(categories) == 1:
                    logger.info(
                        f"Will also plot the SM shapes for other generators"
                    )
                    other_sm_shapes_dicts = [
                        {
                            "shape": sm_shapes_noNNLOPS[observable],
                            "label": "aMC@NLO",
                            "color": "lightcoral",
                            "where": "left"
                        },
                        {
                            "shape": sm_shapes_powheg[observable],
                            "label": "Powheg",
                            "color": "lightseagreen",
                            "where": "right"
                        },
                    ]
                final_plot_output_name = (
                    f"Final{'Asimov' if i == 1 else ''}-{observable}-"
                    + "_".join(categories + singles)
                )
                logger.info(f"Final plot output name: {final_plot_output_name}")

                kappa_spectrum = None
                if args.kappa_prediction:
                    kappa_dct = TK_parser(args.kappa_prediction)
                    kappa_spectrum = ObservableShapeKappa(
                        parameters=kappa_dct["parameters"],
                        edges=kappa_dct["edges"],
                        nominal_values=kappa_dct["crosssection"],
                        ratio_to_sm=kappa_dct["ratio"],
                    )
                    logger.info(f"Kappa prediction: {kappa_dct}")
                    final_plot_output_name += "_kappa"
                final_plot = DiffXSsPerObservable(
                    final_plot_output_name,
                    sm_shapes[observable],
                    shapes,
                    shapes_systonly,
                    other_sm_shapes_dicts=other_sm_shapes_dicts,
                    kappa_prediction=kappa_spectrum,
                )
                final_plot.dump(output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
