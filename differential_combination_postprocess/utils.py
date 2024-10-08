import logging
from rich.logging import RichHandler
import yaml
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
import json


class CustomFilter(logging.Filter):
    def __init__(self, modules=None):
        self.modules = modules if modules else []
        self.modules.extend(["differential_combination_postprocess", "root"])

    def filter(self, record):
        base = record.name.split(".")[0]
        return base in self.modules


def setup_logging(level=logging.INFO):
    logger = logging.getLogger()

    logger.setLevel(level)
    formatter = logging.Formatter("%(message)s")

    stream_handler = RichHandler(show_time=False, rich_tracebacks=True)
    stream_handler.setFormatter(formatter)
    filter = CustomFilter()
    stream_handler.addFilter(filter)
    logger.addHandler(stream_handler)

    return logger


def extract_from_yaml_file(path_to_file):
    with open(path_to_file) as fl:
        dct = yaml.load(fl, Loader=yaml.FullLoader)

    return dct


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def custom_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    cmap = cmap.reversed()
    cmap = truncate_colormap(cmap, 0.3, 1.0, 1000)
    cmap_colors = cmap(np.linspace(minval, maxval, n))
    cmap_colors[-1] = np.array([1, 1, 1, 1])  # Set the highest level to white
    new_cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)
    return new_cmap


def TK_parser(path_to_file):
    dct_to_return = {
        "parameters": {}
    }
    with open(path_to_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 1:
            # line is kappab=0.0
            par, val = line.split("=")
            val = float(val)
            dct_to_return["parameters"][par] = val
        if i == 2:
            # line is kappac=0.0
            par, val = line.split("=")
            val = float(val)
            dct_to_return["parameters"][par] = val
        if line.startswith("binBoundaries"):
            l = line.replace("binBoundaries=", "")
            bin_boundaries = [float(x) for x in l.split(",")]
            dct_to_return["edges"] = bin_boundaries
        if line.startswith("crosssection="):
            l = line.replace("crosssection=", "")
            crosssections = [float(x) for x in l.split(",")]
            # multuply by 1000 to get pb
            crosssections = [x * 1000 for x in crosssections]
            dct_to_return["crosssection"] = crosssections
        if line.startswith("crossecction_integrated="):
            l = line.replace("crossecction_integrated=", "")
            crossecction_integrated = [float(x) for x in l.split(",")]
            # multuply by 1000 to get pb
            crossecction_integrated = [x * 1000 for x in crossecction_integrated]
            dct_to_return["crossecction_integrated"] = crossecction_integrated
        if line.startswith("ratios"):
            l = line.replace("ratios=", "")
            ratios = [float(x) for x in l.split(",")]
            dct_to_return["ratio"] = ratios

    return dct_to_return


# for SMEFT
max_to_matt = {
    "hgg": "gamgam",
    "hzz": "ZZ",
    "htt": "tautau",
    "hww": "WW",
    "hbbvbf": "bb",
    "httboost": "tautau",
}

def mu(pois, coeff_prod, coeff_decay, coeff_tot, fit_model="full"):
    """
    pois: numpy array of pois to be tested
    others can be lists
    """
    if fit_model == "linearised":
        return 1 + pois * (coeff_prod[0] + coeff_decay[0] - coeff_tot[0])
    else:
        if fit_model == "linear":
            prod = 1 + pois * coeff_prod[0]
            decay = 1 + pois * coeff_decay[0]
            tot = 1 + pois * coeff_tot[0]
        else:
            prod = 1 + pois * coeff_prod[0] + pois**2 * coeff_prod[1]
            decay = 1 + pois * coeff_decay[0] + pois**2 * coeff_decay[1]
            tot = 1 + pois * coeff_tot[0] + pois**2 * coeff_tot[1]

        return prod * (decay / tot)


def get_coeffs(poi, production_dct, decays_dct, channel, plus=False, minus=False):
    dec = max_to_matt[channel]
    tot = "tot"
    decay_coeffs = [
        decays_dct[dec][f"A_{poi}"] if f"A_{poi}" in decays_dct[dec] else 0.0,
        decays_dct[dec][f"B_{poi}_2"] if f"B_{poi}_2" in decays_dct[dec] else 0.0,
    ]
    if plus:
        if f"u_A_{poi}" in decays_dct[dec]:
            decay_coeffs[0] += decays_dct[dec][f"u_A_{poi}"]
        if f"u_B_{poi}_2" in decays_dct[dec]:
            decay_coeffs[1] += decays_dct[dec][f"u_B_{poi}_2"]
    if minus:
        if f"u_A_{poi}" in decays_dct[dec]:
            decay_coeffs[0] -= decays_dct[dec][f"u_A_{poi}"]
        if f"u_B_{poi}_2" in decays_dct[dec]:
            decay_coeffs[1] -= decays_dct[dec][f"u_B_{poi}_2"]
    tot_coeffs = [
        decays_dct[tot][f"A_{poi}"] if f"A_{poi}" in decays_dct[tot] else 0.0,
        decays_dct[tot][f"B_{poi}_2"] if f"B_{poi}_2" in decays_dct[tot] else 0.0,
    ]
    if plus:
        if f"u_A_{poi}" in decays_dct[tot]:
            tot_coeffs[0] += decays_dct[tot][f"u_A_{poi}"]
        if f"u_B_{poi}_2" in decays_dct[tot]:
            tot_coeffs[1] += decays_dct[tot][f"u_B_{poi}_2"]
    if minus:
        if f"u_A_{poi}" in decays_dct[tot]:
            tot_coeffs[0] -= decays_dct[tot][f"u_A_{poi}"]
        if f"u_B_{poi}_2" in decays_dct[tot]:
            tot_coeffs[1] -= decays_dct[tot][f"u_B_{poi}_2"]
    production_coeffs = {}
    for k in production_dct:
        prod_coeff = [
            production_dct[k][f"A_{poi}"] if f"A_{poi}" in production_dct[k] else 0.0,
            production_dct[k][f"B_{poi}_2"]
            if f"B_{poi}_2" in production_dct[k]
            else 0.0,
        ]
        if plus:
            if f"u_A_{poi}" in production_dct[k]:
                prod_coeff[0] += production_dct[k][f"u_A_{poi}"]
            if f"u_B_{poi}_2" in production_dct[k]:
                prod_coeff[1] += production_dct[k][f"u_B_{poi}_2"]
        if minus:
            if f"u_A_{poi}" in production_dct[k]:
                prod_coeff[0] -= production_dct[k][f"u_A_{poi}"]
            if f"u_B_{poi}_2" in production_dct[k]:
                prod_coeff[1] -= production_dct[k][f"u_B_{poi}_2"]
        production_coeffs[k] = prod_coeff

    return production_coeffs, decay_coeffs, tot_coeffs

full_production_files = {
    "smH_PTH": {
        "hgg": "{}/differentials/hgg/FullProduction_pt_h.json",
        "hzz": "{}/differentials/hzz/FullProduction_pt_h.json",
        "htt": "{}/differentials/htt/FullProduction_pt_h.json",
        "hww": "{}/differentials/hww/FullProduction_pt_h.json",
        "hbbvbf": "{}/differentials/hbbvbf/FullProduction_pt_h.json",
        "httboost": "{}/differentials/httboost/FullProduction_pt_h.json",
    },
    "DeltaPhiJJ": {
        "hgg": "{}/differentials/hgg/FullProduction_deltaphijj.json",
        "hzz": "{}/differentials/hzz/FullProduction_deltaphijj.json",
    },
}

def refactor_predictions(prediction_dir, channel, observable):
    decays_file = f"{prediction_dir}/decay.json"
    with open(decays_file, "r") as f:
        decays_dct = json.load(f)
    # production_file = ggH_production_files[observable][channel].format(prediction_dir)
    production_file = full_production_files[observable][channel].format(prediction_dir)
    with open(production_file, "r") as f:
        tmp_production_dct = json.load(f)
    dict_keys = list(tmp_production_dct.keys())
    sorted_keys = sorted(dict_keys, key=lambda x: float(x))
    tmp_production_dct = {k: tmp_production_dct[k] for k in sorted_keys}
    production_dct = {}
    if observable == "smH_PTH":
        edges = [float(k) for k in sorted_keys] + [10000.0]
        for edge, next_edge in zip(edges[:-1], edges[1:]):
            production_dct[
                "r_smH_PTH_{}_{}".format(
                    str(edge).replace(".0", ""), str(next_edge).replace(".0", "")
                )
            ] = tmp_production_dct[str(edge)]
        if channel in ["hgg", "htt"]:
            key_to_remove = "r_smH_PTH_450_10000"
            key_dct = production_dct[key_to_remove]
            production_dct.pop(key_to_remove)
            production_dct["r_smH_PTH_GT450"] = key_dct
        elif channel in ["hzz", "hww"]:
            key_to_remove = "r_smH_PTH_200_10000"
            key_dct = production_dct[key_to_remove]
            production_dct.pop(key_to_remove)
            production_dct["r_smH_PTH_GT200"] = key_dct
        elif channel == "httboost":
            key_to_remove = "r_smH_PTH_600_10000"
            key_dct = production_dct[key_to_remove]
            production_dct.pop(key_to_remove)
            production_dct["r_smH_PTH_GT600"] = key_dct
        elif channel == "hbbvbf":
            # this is we don't add manually 800 to the prediction JSON
            #key_to_remove = "r_smH_PTH_675_10000"
            #key_dct = production_dct[key_to_remove]
            #production_dct.pop(key_to_remove)
            #production_dct["r_smH_PTH_675_800"] = key_dct
            #edges[-1] = 800.0
            key_to_remove = "r_smH_PTH_800_10000"
            key_dct = production_dct[key_to_remove]
            production_dct.pop(key_to_remove)
            production_dct["r_smH_PTH_800_1200"] = key_dct
    elif observable == "Njets":
        convert = {
            "-0.5": "r_Njets_0",
            "0.5": "r_Njets_1",
            "1.5": "r_Njets_2",
            "2.5": "r_Njets_3",
            "3.5": "r_Njets_4",
        }
        edges = [float(k) for k in sorted_keys] + [4.5]
        for old, new in convert.items():
            production_dct[new] = tmp_production_dct[old]
    elif observable == "DeltaPhiJJ":
        edges = [float(k) for k in sorted_keys] + [3.16]
        for edge, next_edge in zip(edges[:-1], edges[1:]):
            production_dct[
                "r_DeltaPhiJJ_{}_{}".format(
                    str(edge).replace(".0", ""), str(next_edge).replace(".0", "")
                )
            ] = tmp_production_dct[str(edge)]

    return decays_dct, production_dct, edges, sorted_keys

def refactor_predictions_multichannel(prediction_dir, config):
    # config is a dictionary {channel: observable}
    production_dct = {}
    edges = {}
    for channel, observable in config.items():
        decays_dct, production_dct[channel], edges[channel], _ = refactor_predictions(
            prediction_dir, channel, observable
        )
    return decays_dct, production_dct, edges
