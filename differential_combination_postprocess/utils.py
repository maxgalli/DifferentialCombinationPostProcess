import logging
from rich.logging import RichHandler
import yaml
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt


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