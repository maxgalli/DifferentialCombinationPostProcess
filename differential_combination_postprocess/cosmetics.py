from matplotlib import cm
import numpy as np

np.random.seed(0)

markers = ("v", "^", ">", "<", "s", "p", "*")

rainbow = cm.rainbow(np.linspace(0, 1, 20))
np.random.shuffle(rainbow)

fit_type_colors = ["red", "green", "blue", "purple"]

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
    "yH": {
        "x_plot_label": "$y_{H}$",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta y_{H}$ [fb]",
    },
    "smH_PTJ0": {
        "x_plot_label": "$p_{T}^{jet}$ [GeV]",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta p_{T}^{jet}$ [fb/GeV]",
    },
}

category_specs = {
    "Hgg": {
        "plot_label": "H $\longrightarrow\\gamma\\gamma$",
        "color": "red",
        "marker": "^",
    },
    "HZZ": {"plot_label": "H $\longrightarrow$ ZZ", "color": "blue", "marker": "v"},
    "HWW": {"plot_label": "H $\longrightarrow$ WW", "color": "purple", "marker": "s"},
    "Htt": {
        "plot_label": "H $\longrightarrow\\tau \\tau$",
        "color": "green",
        "marker": "p",
    },
    "Hbb": {"plot_label": "H $\longrightarrow$ bb", "color": "orange", "marker": "*"},
    "HbbVBF": {
        "plot_label": "H $\longrightarrow$ bb",
        "color": "orange",
        "marker": "*",
    },
    "HttBoost": {
        "plot_label": "H $\longrightarrow\\tau \\tau$ Boosted",
        "color": "hotpink",
        "marker": "D",
    },
    "HggHZZ": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHWW": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHZZHWW": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHWWHtt": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHZZHWWHtt": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHZZHWWHttHbb": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHZZHWWHttHbbVBF": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
}

TK_parameters_labels = {
    "kappab": "$\kappa_{b}$",
    "kappac": "$\kappa_{c}$",
    "ct": "$\kappa_{t}$",
    "cg": "$c_{g}$",
    "cb": "$\kappa_{b}$",
}

SMEFT_parameters_labels = {
    "chg": "$c_{HG}$",
    "chgtil": "$c_{H\widetilde{G}}$",
    "chb": "$c_{HB}$",
    "chbtil": "$c_{H\widetilde{B}}$",
    "chw": "$c_{HW}$",
    "chwtil": "$c_{H\widetilde{W}}$",
    "chwb": "$c_{HWB}$",
    "chwbtil": "$c_{H\widetilde{W}B}$",
    "EV0": "$EV_{0}$",
    "EV1": "$EV_{1}$",
    "EV2": "$EV_{2}$",
    "EV3": "$EV_{3}$",
    "EV4": "$EV_{4}$",
    "EV5": "$EV_{5}$",
}
