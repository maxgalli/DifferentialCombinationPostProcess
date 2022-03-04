from matplotlib import cm
import numpy as np

np.random.seed(0)

markers = ("v", "^", ">", "<", "s", "p", "*")

rainbow = cm.rainbow(np.linspace(0, 1, 20))
np.random.shuffle(rainbow)

# All the minor decorations for the final plot which depend on the observable itself

observable_specs = {
    "smH_PTH": {
        "x_plot_label": "$p_{T}^{H}$ [GeV]",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta p_{T}^{H}$ [fb/GeV]",
    },
    "Njets": {
        "x_plot_label": "$N_{jets}$",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta N_{jets}$ [fb]",
    },
}

category_specs = {
    "Hgg": {
        "plot_label": "H $\longrightarrow\\gamma\\gamma$",
        "color": "red",
        "marker": "^",
    },
    "HZZ": {"plot_label": "H $\longrightarrow$ ZZ", "color": "blue", "marker": "v"},
}
