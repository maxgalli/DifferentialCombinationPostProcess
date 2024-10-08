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
        "x_plot_label": "$p_{T}^{H}$ (GeV)",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta p_{T}^{H}$ (fb/GeV)",
    },
    "Njets": {
        "x_plot_label": "$N_{jets}$",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta N_{jets}$ (fb)",
    },
    "yH": {
        "x_plot_label": "$|y_{H}|$",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta |y_{H}|$ (fb)",
    },
    "smH_PTJ0": {
        "x_plot_label": "$p_{T}^{jet}$ (GeV)",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta p_{T}^{jet}$ (fb/GeV)",
    },
    "mjj": {
        "x_plot_label": "$m_{jj}$ (GeV)",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta m_{jj}$ (fb/GeV)",
    },
    "DEtajj": {
        "x_plot_label": "$\\Delta\\eta_{jj}$",
        "y_plot_label": "$\\Delta\\sigma\ / \\Delta\\eta_{jj}$ (fb)",
    },
    "TauCJ": {
        "x_plot_label": "$\\tau_{C}^{j}$ (GeV)",
        "y_plot_label": "$\\Delta\\sigma\ / \Delta \\tau_{C}^{j}$ (fb/GeV)",
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
    "HttOriginal": {
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
    "HggHZZHtt": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHZZHttBoost": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHWWHtt": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHZZHWWHtt": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHZZHWWHttHttBoost": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
    "HggHZZHWWHttHbb": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "HggHZZHWWHttHbbVBF": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
    "HggHZZHttHttBoostHbbVBF": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
    "HggHZZHttHttBoost": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
    "HggHZZHWWHttHbbVBFHttBoost": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
    "FinalComb": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
    "FinalComb2": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
    "comb": {
        "plot_label": "Combination",
        "color": "black",
        "marker": "o",
    },
    "comb_asimov": {
        "plot_label": "Combination Expected",
        "color": "orange",
        "marker": "o",
    },
    "HggHZZHtt_asimov": {
        "plot_label": "Combination Expected",
        "color": "red",
        "marker": "o",
    },
    # SMEFT
    "PtFullComb": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "PtFullComb_asimov": {"plot_label": "Combination expected", "color": "red", "marker": "o"},
    "PtFullComb2": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "PtFullComb2_asimov": {"plot_label": "Combination expected", "color": "red", "marker": "o"},
    "ChanObs4": {"plot_label": "Combination", "color": "black", "marker": "o"},
    "PtHgg": {
        "plot_label": "H $\longrightarrow\\gamma\\gamma$",
        "color": "red",
        "marker": "^",
    },
    "PtHggHZZ": {
        "plot_label": "H $\longrightarrow\\gamma\\gamma$ + H $\longrightarrow$ ZZ",
        "color": "black",
        "marker": "o",
    },
    "DeltaPhiJJHggHZZ": {
        "plot_label": "H $\longrightarrow\\gamma\\gamma$ + H $\longrightarrow$ ZZ",
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
    "kappa_lambda": "$\kappa_{\lambda}$",
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
    "chdd": "$c_{HD}$",
    "chd": "$c_{Hd}$",
    "cll1": "$c^{(\prime)}_{ll}$",
    "chl1": "$c^{(1)}_{Hl}$",
    "chl3": "$c^{(3)}_{Hl}$",
    "chq1": "$c^{(1)}_{HQ}$",
    "chq3": "$c^{(3)}_{HQ}$",
    "chj1": "$c^{(1)}_{Hq}$",
    "chj3": "$c^{(3)}_{Hq}$",
    "chu": "$c_{Hu}$",
    "chbox": "$c_{Hbox}$",
    "cw": "$c_{W}$",
    "cbhre": "$Re(c_{bH})$",
    "cbhim": "$Im(c_{bH})$",
    "cthre": "$Re(c_{tH})$",
    "cthim": "$Im(c_{tH})$",
    "ctgre": "$Re(c_{tG})$",
    "cbbre": "$Re(c_{bB})$",
    "cbwim": "$Im(c_{bW})$",
    "cbwre": "$Re(c_{bW})$",
    "cehre": "$Re(c_{eH})$",
    "cehim": "$Im(c_{eH})$",
    "che": "$c_{He}$",
    "chbq": "$c_{Hb}$",
    "ctbre": "$Re(c_{tB})$",
    "cht": "$c_{Ht}$",
    "ctwre": "$Re(c_{tW})$",
    "cuhre": "$Re(c_{uH})$",
}

for i in range(100):
    SMEFT_parameters_labels[f"EV{i}"] = "$EV_{" + f"{i}" + "}$"

matrix_bin_names = {
    "r_smH_PTH_0_5": "0 - 5",
    "r_smH_PTH_5_10": "5 - 10",
    "r_smH_PTH_10_15": "10 - 15",
    "r_smH_PTH_15_20": "15 - 20",
    "r_smH_PTH_20_25": "20 - 25",
    "r_smH_PTH_25_30": "25 - 30",
    "r_smH_PTH_30_35": "30 - 35",
    "r_smH_PTH_35_45": "35 - 45",
    "r_smH_PTH_45_60": "45 - 60",
    "r_smH_PTH_60_80": "60 - 80",
    "r_smH_PTH_80_100": "80 - 100",
    "r_smH_PTH_100_120": "100 - 120",
    "r_smH_PTH_120_140": "120 - 140",
    "r_smH_PTH_140_170": "140 - 170",
    "r_smH_PTH_170_200": "170 - 200",
    "r_smH_PTH_200_250": "200 - 250",
    "r_smH_PTH_250_350": "250 - 350",
    "r_smH_PTH_350_450": "350 - 450",
    "r_smH_PTH_450_500": "450 - 500",
    "r_smH_PTH_500_550": "500 - 550",
    "r_smH_PTH_550_600": "550 - 600",
    "r_smH_PTH_600_675": "600 - 675",
    "r_smH_PTH_675_800": "675 - 800",
    "r_smH_PTH_800_1200": "> 800",
    "r_smH_PTH_GT450": "> 450",
    "r_smH_PTJ0_0_30": "0 - 30",
    "r_smH_PTJ0_30_40": "30 - 40",
    "r_smH_PTJ0_40_55": "40 - 55",
    "r_smH_PTJ0_55_75": "55 - 75",
    "r_smH_PTJ0_75_95": "75 - 95",
    "r_smH_PTJ0_95_120": "95 - 120",
    "r_smH_PTJ0_120_150": "120 - 150",
    "r_smH_PTJ0_150_200": "150 - 200",
    "r_smH_PTJ0_200_450": "200 - 450",
    "r_smH_PTJ0_450_600": "450 - 600",
    "r_smH_PTJ0_GT600": "> 600",
    "r_smH_PTJ0_GT200": "> 200",
    "r_smH_PTH_0_45": "0 - 45",
    "r_smH_PTH_45_80": "45 - 80",
    "r_smH_PTH_80_120": "80 - 120",
    "r_smH_PTH_200_350": "200 - 350",
    "r_smH_PTH_0_10": "0 - 10",
    "r_smH_PTH_10_20": "10 - 20",
    "r_smH_PTH_20_30": "20 - 30",
    "r_smH_PTH_30_45": "30 - 45",
    "r_smH_PTH_120_200": "120 - 200",
    "r_smH_PTH_GT200": "> 200",
    "r_smH_PTH_450_600": "450 - 600",
    "r_smH_PTH_GT600": "> 600",
    "r_Njets_0": "0",
    "r_Njets_1": "1",
    "r_Njets_2": "2",
    "r_Njets_3": "3",
    "r_Njets_4": ">=4",
    "r_yH_0p0_0p15": "0 - 0.15",
    "r_yH_0p15_0p3": "0.15 - 0.3",
    "r_yH_0p3_0p45": "0.3 - 0.45",
    "r_yH_0p45_0p6": "0.45 - 0.6",
    "r_yH_0p6_0p75": "0.6 - 0.75",
    "r_yH_0p75_0p9": "0.75 - 0.9",
    "r_yH_0p9_1p2": "0.9 - 1.2",
    "r_yH_1p2_1p6": "1.2 - 1.6",
    "r_yH_1p6_2p0": "1.6 - 2.0",
    "r_yH_2p0_2p5": "2.0 - 2.5",
    "r_DEtajj_out": "out",
    "r_DEtajj_0p0_0p7": "0 - 0.7",
    "r_DEtajj_0p7_1p6": "0.7 - 1.6",
    "r_DEtajj_1p6_3p0": "1.6 - 3.0",
    "r_DEtajj_3p0_5p0": "3.0 - 5.0",
    "r_DEtajj_GT5p0": "> 5.0",
    "r_mjj_out": "out",
    "r_mjj_0_75": "0 - 75",
    "r_mjj_75_120": "75 - 120",
    "r_mjj_120_180": "120 - 180",
    "r_mjj_180_300": "180 - 300",
    "r_mjj_300_500": "300 - 500",
    "r_mjj_500_1000": "500 - 1000",
    "r_mjj_GT1000": "> 1000",
    "r_TauCJ_out": "out",
    "r_TauCJ_out_hzz": "out ZZ",
    "r_TauCJ_15_20": "15 - 20",
    "r_TauCJ_20_30": "20 - 30",
    "r_TauCJ_30_50": "30 - 50",
    "r_TauCJ_50_80": "50 - 80",
    "r_TauCJ_GT80": "> 80",
}

bin_names = {
    "hgg_0p0_5p0": "H $\longrightarrow\\gamma\\gamma$, 0-5",
    "hgg_5p0_10p0": "H $\longrightarrow\\gamma\\gamma$, 5-10",
    "hgg_10p0_15p0": "H $\longrightarrow\\gamma\\gamma$, 10-15",
    "hgg_15p0_20p0": "H $\longrightarrow\\gamma\\gamma$, 15-20",
    "hgg_20p0_25p0": "H $\longrightarrow\\gamma\\gamma$, 20-25",
    "hgg_25p0_30p0": "H $\longrightarrow\\gamma\\gamma$, 25-30",
    "hgg_30p0_35p0": "H $\longrightarrow\\gamma\\gamma$, 30-35",
    "hgg_35p0_45p0": "H $\longrightarrow\\gamma\\gamma$, 35-45",
    "hgg_45p0_60p0": "H $\longrightarrow\\gamma\\gamma$, 45-60",
    "hgg_60p0_80p0": "H $\longrightarrow\\gamma\\gamma$, 60-80",
    "hgg_80p0_100p0": "H $\longrightarrow\\gamma\\gamma$, 80-100",
    "hgg_100p0_120p0": "H $\longrightarrow\\gamma\\gamma$, 100-120",
    "hgg_120p0_140p0": "H $\longrightarrow\\gamma\\gamma$, 120-140",
    "hgg_140p0_170p0": "H $\longrightarrow\\gamma\\gamma$, 140-170",
    "hgg_170p0_200p0": "H $\longrightarrow\\gamma\\gamma$, 170-200",
    "hgg_200p0_250p0": "H $\longrightarrow\\gamma\\gamma$, 200-250",
    "hgg_250p0_350p0": "H $\longrightarrow\\gamma\\gamma$, 250-350",
    "hgg_350p0_450p0": "H $\longrightarrow\\gamma\\gamma$, 350-450",
    "hgg_450p0_10000p0": "H $\longrightarrow\\gamma\\gamma$, > 450",
    "hzz_0p0_10p0": "H $\longrightarrow$ ZZ, 0-10",
    "hzz_10p0_20p0": "H $\longrightarrow$ ZZ, 10-20",
    "hzz_20p0_30p0": "H $\longrightarrow$ ZZ, 20-30",
    "hzz_30p0_45p0": "H $\longrightarrow$ ZZ, 30-45",
    "hzz_45p0_60p0": "H $\longrightarrow$ ZZ, 45-60",
    "hzz_60p0_80p0": "H $\longrightarrow$ ZZ, 60-80",
    "hzz_80p0_120p0": "H $\longrightarrow$ ZZ, 80-120",
    "hzz_120p0_200p0": "H $\longrightarrow$ ZZ, 120-200",
    "hzz_200p0_10000p0": "H $\longrightarrow$ ZZ, > 200",
    "hww_0p0_30p0": "H $\longrightarrow$ WW, 0-30",
    "hww_30p0_45p0": "H $\longrightarrow$ WW, 30-45",
    "hww_45p0_80p0": "H $\longrightarrow$ WW, 45-80",
    "hww_80p0_120p0": "H $\longrightarrow$ WW, 80-120",
    "hww_120p0_200p0": "H $\longrightarrow$ WW, 120-200",
    "hww_200p0_10000p0": "H $\longrightarrow$ WW, > 200",
    "htt_0p0_45p0": "H $\longrightarrow\\tau \\tau$, 0-45",
    "htt_45p0_80p0": "H $\longrightarrow\\tau \\tau$, 45-80",
    "htt_80p0_120p0": "H $\longrightarrow\\tau \\tau$, 80-120",
    "htt_120p0_140p0": "H $\longrightarrow\\tau \\tau$, 120-140",
    "htt_140p0_170p0": "H $\longrightarrow\\tau \\tau$, 140-170",
    "htt_170p0_200p0": "H $\longrightarrow\\tau \\tau$, 170-200",
    "htt_200p0_350p0": "H $\longrightarrow\\tau \\tau$, 200-350",
    "htt_350p0_450p0": "H $\longrightarrow\\tau \\tau$, 350-450",
    "htt_450p0_10000p0": "H $\longrightarrow\\tau \\tau$, > 450",
    "httboost_450p0_600p0": "H $\longrightarrow\\tau \\tau$ boosted, 450-600",
    "httboost_600p0_10000p0": "H $\longrightarrow\\tau \\tau$ boosted, > 600",
}

def get_parameter_label(poi):
    try:
        return TK_parameters_labels[poi]
    except KeyError:
        try:
            return SMEFT_parameters_labels[poi]
        except KeyError:
            return poi