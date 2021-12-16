import logging
from rich.logging import RichHandler
import yaml


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


def merge_bins(old_bins, old_edges, new_edges):
    if not all(edge in old_edges for edge in new_edges):
        raise ValueError("New edges are not a subset of old edges")

    # Loop oever pairs of consecutive edges in new_edges
    new_bins = []
    for first, second in zip(new_edges, new_edges[1:]):
        old_first_index = old_edges.index(first)
        old_second_index = old_edges.index(second)
        new_bins.append(sum(old_bins[old_first_index:old_second_index]))

    if len(new_bins) != len(new_edges) - 1:
        raise ValueError("Something went wrong")

    return new_bins