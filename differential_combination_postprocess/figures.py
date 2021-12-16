import matplotlib.pyplot as plt
import mplhep as hep
hep.set_style("CMS")

from .cosmetics import rainbow_iter
from .shapes import ObservableShapeSM

# Silence matplotlib warnings for Christ sake
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import logging
logger = logging.getLogger(__name__)


class Figure:
    def __init__(self):
        self.fig, _ = plt.subplots()
        self.output_name = "prototype"

    def dump(self, output_dir):
        # Dump the image in multiple formats
        self.fig.savefig("{}/{}.png".format(output_dir, self.output_name), bbox_inches="tight")
        self.fig.savefig("{}/{}.pdf".format(output_dir, self.output_name), bbox_inches="tight")


class XSNLLsPerPOI:
    """
    """
    def __init__(self, subcategory_spectra):
        self.figures = []
        nominal_spectrum = list(subcategory_spectra.values())[0]
        pois = nominal_spectrum.scans.keys()
        
        for poi in pois:
            scans = []
            for subcategory, spectrum in subcategory_spectra.items():
                scans.append((subcategory, spectrum.scans[poi]))
            
            fig, ax = plt.subplots()
            output_name = f"NLLs_{nominal_spectrum.variable}_{nominal_spectrum.category}_{poi}"

            # Set labels
            ax.set_xlabel(poi)
            ax.set_ylabel("-2$\Delta$lnL")

            # Set limits
            ax.set_ylim(0., 4.)

            # Draw horizontal line at 1
            ax.axhline(1., color="k", linestyle="--")

            # Draw all the NLLs with different colors
            for scan_tpl, color in zip(scans, rainbow_iter):
                ax = scan_tpl[1].plot(ax, color, label=scan_tpl[0])

            # Legend
            ax.legend(loc='upper center', prop={'size': 10}, ncol=4)
            hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=35.9, ax=ax)

            self.figures.append((fig, ax, output_name))

    def dump(self, output_dir):
        for fig, ax, output_name in self.figures:
            # Dump the image in multiple formats
            fig.savefig("{}/{}.png".format(output_dir, output_name), bbox_inches="tight")
            fig.savefig("{}/{}.pdf".format(output_dir, output_name), bbox_inches="tight")


class XSNLLsPerCategory(Figure):
    """ Plot the NLLs for a given category, one NLL per POI
    """
    def __init__(self, differential_spectrum):
        self.ds = differential_spectrum
        self.fig, self.ax = plt.subplots()
        self.output_name = "NLLs_{}_{}".format(
            self.ds.variable, self.ds.category
            )

        # Set labels
        self.ax.set_xlabel(self.ds.variable)
        self.ax.set_ylabel("-2$\Delta$lnL")

        # Set limits
        self.ax.set_ylim(0., 4.)

        # Draw horizontal line at 1
        self.ax.axhline(1., color="k", linestyle="--")
        
        # Draw all the NLLs on the ax
        logger.debug(differential_spectrum.scans)
        for poi_scan, color in zip(differential_spectrum.scans.items(), rainbow_iter):
            poi, scan = poi_scan
            self.ax = scan.plot(self.ax, color)

        # Legend
        self.ax.legend(loc='upper center', prop={'size': 10}, ncol=4)
        hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=35.9, ax=self.ax)


class DiffXSsPerObservable(Figure):
    """
    """
    def __init__(self, sm_shape, observable_shapes):
        self.output_name = "prova"
        # Set up figure and axes
        self.fig, (self.main_ax, self.ratio_ax) = plt.subplots(
            nrows=2,
            ncols=1,
            gridspec_kw={"height_ratios": (3, 1)},
            sharex=True
            )

        # X limits depend on the SM plot (the one in the background with the predictions)
        self.main_ax.set_xlim(sm_shape.fake_edges[0], sm_shape.fake_edges[-1])
        self.ratio_ax.set_xlim(sm_shape.fake_edges[0], sm_shape.fake_edges[-1])
        self.main_ax.set_yscale("log")
        self.ratio_ax.set_ylim(-2, 4)
        self.main_ax, self.ratio_ax = sm_shape.plot(self.main_ax, self.ratio_ax)

        for shape, color in zip(observable_shapes, rainbow_iter):
            shape.stretch_fake_range(sm_shape)
            self.main_ax, self.ratio_ax = shape.plot(self.main_ax, self.ratio_ax, color)

        hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=35.9, ax=self.main_ax)