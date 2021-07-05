import matplotlib.pyplot as plt
import mplhep as hep
hep.set_style("CMS")

from .cosmetics import black_to_grey, rainbow

# Silence matplotlib warnings for Christ sake
import warnings
warnings.filterwarnings("ignore", module="matplotlib")


class Figure:
    def __init__(self):
        self.fig, _ = plt.subplots()
        self.output_name = "prototype"

    def dump(self, output_dir):
        # Dump the image in multiple formats
        self.fig.savefig("{}/{}.png".format(output_dir, self.output_name), bbox_inches="tight")
        self.fig.savefig("{}/{}.pdf".format(output_dir, self.output_name), bbox_inches="tight")


class XSNLLsPerPOI(Figure):
    """
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
        color_index = 0
        for poi, scan in differential_spectrum.scans.items():
            self.ax = scan.plot(self.ax, rainbow[color_index])
            color_index += 1

        # Legend
        self.ax.legend()
        hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=35.9, ax=self.ax)