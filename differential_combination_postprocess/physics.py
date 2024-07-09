import numpy as np
import scipy.interpolate as itr
import pickle as pkl
import awkward as ak
import os

import logging

logger = logging.getLogger(__name__)

"""
With stuff received from Thomas
"""


def get_prediction(arr, mass, weights=None, interPRepl=None, massRepl=None):
    # Defined by Thomas in https://github.com/threiten/HiggsAnalysis-CombinedLimit/blob/d5d9ef377a7c69a8d4eaa366b47e7c81931e71d9/test/plotBinnedSigStr.py#L236
    # To be used with weights [1, 2.3, 1]
    nBins = len(arr)
    masses = [120.0, 125.0, 130.0]
    if weights is None:
        weights = np.ones(arr.shape[1])
    splines = []
    if arr.shape[1] == 1:
        if interPRepl is None or massRepl is None:
            raise ValueError(
                "If only one masspoint is given, interPRepl and massRepl must be provided!"
            )
        for i in range(nBins):
            splines.append(
                itr.UnivariateSpline(masses, interPRepl[i, :], w=weights, k=2)
            )

        return np.array(
            [
                splines[i](mass) - interPRepl[i, masses.index(massRepl)] + arr[i, 0]
                for i in range(nBins)
            ]
        )

    for i in range(nBins):
        splines.append(itr.UnivariateSpline(masses, arr[i, :], w=weights, k=2))

    return np.array([splines[i](mass) for i in range(nBins)])


# Everything poorly hardcoded, will change later

theor_pred_base_dir = "/work/gallim/DifferentialCombination_home/DiffCombOrchestrator/DifferentialCombinationRun2/TheoreticalPredictions/231003"
mass = 125.38
weights = [1.0, 2.3, 1.0]
hgg_br = 0.0023

analyses_edges = {
    "smH_PTH": {
        "Hgg": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            #1000,
            1200,
        ],
        "HZZ": [
            0,
            10,
            20,
            30,
            45,
            60,
            80,
            120,
            200,
            #1000,
            1200,
        ],  # because we merged last three bins
        "HWW": [0, 30, 45, 80, 120, 200, 1200],
        "Htt": [0, 45, 80, 120, 140, 170, 200, 350, 450, 1200],
        "HttOriginal": [0, 45, 80, 120, 200, 350, 450, 1200],
        "Hbb": [300, 450, 650, 1200],
        "HbbVBF": [450, 500, 550, 600, 675, 800, 1200],
        "HttBoost": [450, 600, 1200],
        "HggHZZ": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            1200,
        ],
        "HggHZZHWW": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            1000,
        ],
        "HggHZZHWWHtt": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            1000,
        ],
        "HggHZZHWWHttHbb": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            650,
            1000,
        ],
        "HggHZZHWWHttHbbVBF": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            500,
            550,
            600,
            675,
            800,
            1200,
        ],
        "HggHZZHWWHttHbbVBFHttBoost": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            500,
            550,
            600,
            675,
            800,
            1200,
        ],
        "MaximumGranularity": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            1000,
        ],
        "EvenMoreMaximumGranularity": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            675,
            700,
            750,
            800,
            1200,
        ],
        "EvenMoreMaximumGranularity4Bkg": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            600,
            1200,
        ],
        "FinalComb": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            #1000,
            1200
        ],
        "FinalComb2": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            45,
            60,
            80,
            100,
            120,
            140,
            170,
            200,
            250,
            350,
            450,
            #1000,
            1200
        ],
    },
    "Njets": {
        "Hgg": [0, 1, 2, 3, 4, 5],
        "HZZ": [0, 1, 2, 3, 4, 5],
        "HWW": [0, 1, 2, 3, 4, 5],
        "Htt": [0, 1, 2, 3, 4, 5],
        "HggHWW": [0, 1, 2, 3, 4, 5],
        "HggHWWHtt": [0, 1, 2, 3, 4, 5],
        "HggHZZHWWHtt": [0, 1, 2, 3, 4, 5],
    },
    "yH": {
        "Hgg": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.20, 1.60, 2.0, 2.5],
        "HZZ": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.20, 1.60, 2.5],
        "HggHZZ": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.20, 1.60, 2.0, 2.5],
        "Granular": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.20, 1.60, 2.0, 2.5],
    },
    "smH_PTJ0": {
        "Hgg": [0.0, 30.0, 40.0, 55.0, 75.0, 95.0, 120.0, 150.0, 200.0, 10000.0],
        "HZZ": [0.0, 30.0, 55.0, 95.0, 200.0, 10000.0],
        "HttBoost": [450.0, 600.0, 10000.0],
        "FinalComb": [0.0, 30.0, 40.0, 55.0, 75.0, 95.0, 120.0, 150.0, 200.0, 10000.0],
        "HggHZZHttBoost": [
            0.0,
            30.0,
            40.0,
            55.0,
            75.0,
            95.0,
            120.0,
            150.0,
            200.0,
            450.0,
            600.0,
            10000.0,
        ],
        "Granular": [
            0.0,
            30.0,
            40.0,
            55.0,
            75.0,
            95.0,
            120.0,
            150.0,
            200.0,
            450.0,
            600.0,
            10000.0,
        ],
    },
    "mjj": {
        "Hgg": [
            0.0,
            75.0,
            120.0,
            180.0,
            300.0,
            500.0,
            1000.0,
            1500.0,
        ],  # 13000 is a because as usual Thomas predictions do this
        "HZZ": [0.0, 120.0, 300.0, 1500.0],
        "HggHZZ": [0.0, 75.0, 120.0, 180.0, 300.0, 500.0, 1000.0, 1500.0],
    },
    "DEtajj": {
        "Hgg": [0.0, 0.7, 1.6, 3.0, 5.0, 100.0],
        "HZZ": [0.0, 1.6, 3.0, 100.0],
        "HggHZZ": [0.0, 0.7, 1.6, 3.0, 5.0, 100.0],
    },
    "TauCJ": {
        "Hgg": [15.0, 20.0, 30.0, 50.0, 80.0, 10000.0],
        "HZZ": [15.0, 20.0, 30.0, 50.0, 80.0, 10000.0],
        "HggHZZ": [15.0, 20.0, 30.0, 50.0, 80.0, 10000.0],
    },
}

overflows = ["smH_PTH", "smH_PTJ0", "mjj", "DEtajj", "TauCJ"]


def make_hgg_theory_pred_array(pickle_central, pickle_uncertainty):
    """
    smH_PTH_xs will be an awkward array with fields 'central', 'up', and 'down'
    Units of measure of Thomas files are fb/bin_width
    N.B.: we normalize it here?, ask Thomas if this is correct!
    also note how this is done for up and down
    """
    obs_xs_dict = {}
    if pickle_central.startswith("/"):
        full_path = pickle_central
    else:
        full_path = os.path.join(theor_pred_base_dir, pickle_central)
    if pickle_uncertainty.startswith("/"):
        full_path_unc = pickle_uncertainty
    else:
        full_path_unc = os.path.join(theor_pred_base_dir, pickle_uncertainty)
    with open(full_path, "rb") as f:
        obs = pkl.load(f)
        if obs.shape[1] == 1:
            obs = np.repeat(obs, 3, axis=1)
        obs = get_prediction(obs, mass, weights=weights)
    obs_xs_dict["central"] = obs
    with open(full_path_unc, "rb") as f:
        obs_uncs = pkl.load(f)
        obs_down = obs - obs_uncs[0, :]
        obs_up = obs + obs_uncs[1, :]
    obs_xs_dict["up"] = obs_up
    obs_xs_dict["down"] = obs_down
    obs_xs = ak.Array(obs_xs_dict)
    for field in obs_xs.fields:
        obs_xs[field] = obs_xs[field] / hgg_br

    return obs_xs


smH_PTH_Hgg_xs = make_hgg_theory_pred_array(
    "theoryPred_Pt_2018_ggHxH_NNLOPS.pkl", "theoryPred_Pt_2018_ggHxH_NNLOPS_theoryUnc.pkl"
)
smH_PTH_MaximumGranularity_xs = make_hgg_theory_pred_array(
    "/work/gallim/DifferentialCombination_home/DifferentialCombinationRun2/TheoreticalPredictions/production_modes/ggH_MoreGranular/theoryPred_Pt_2018_ggHMoreGranular.pkl",
    "/work/gallim/DifferentialCombination_home/DifferentialCombinationRun2/TheoreticalPredictions/production_modes/ggH_MoreGranular/theoryPred_Pt_2018_ggHMoreGranular_theoryUnc.pkl",
)
smH_PTH_EvenMoreMaximumGranularity_xs = make_hgg_theory_pred_array(
    "theoryPred_Pt_2018_ggHxH_EvenMoreGranular_NNLOPS.pkl", "theoryPred_Pt_2018_ggHxH_EvenMoreGranular_NNLOPS_theoryUnc.pkl"
)
Njets_Hgg_xs = make_hgg_theory_pred_array(
    "theoryPred_Njets_2018_ggHxH_NNLOPS.pkl", "theoryPred_Njets_2018_ggHxH_NNLOPS_theoryUnc.pkl"
)
yH_Hgg_xs = make_hgg_theory_pred_array(
    "/work/gallim/DifferentialCombination_home/DiffCombOrchestrator/DifferentialCombinationRun2/TheoreticalPredictions/fullPSPred/theoryPred_AbsRapidityFine_18_fullPS.pkl",
    "/work/gallim/DifferentialCombination_home/DiffCombOrchestrator/DifferentialCombinationRun2/TheoreticalPredictions/fullPSPred/theoryPred_AbsRapidityFine_18_fullPS_theoryUnc.pkl",
)
yH_Granular_xs = make_hgg_theory_pred_array(
    "theoryPred_yH_2018_ggHxH_Granular_NNLOPS.pkl", "theoryPred_yH_2018_ggHxH_Granular_NNLOPS_theoryUnc.pkl"
)
smH_PTJ0_Hgg_xs = make_hgg_theory_pred_array(
    #"/work/gallim/DifferentialCombination_home/DiffCombOrchestrator/DifferentialCombinationRun2/TheoreticalPredictions/fullPSPred/theoryPred_Jet2p5Pt0_18_fullPS.pkl", 
    #"/work/gallim/DifferentialCombination_home/DiffCombOrchestrator/DifferentialCombinationRun2/TheoreticalPredictions/fullPSPred/theoryPred_Jet2p5Pt0_18_fullPS_theoryUnc.pkl"
    "theoryPred_smH_PTJ0_fineBinning_2018_ggHxH_NNLOPS.pkl", "theoryPred_smH_PTJ0_fineBinning_2018_ggHxH_NNLOPS_theoryUnc.pkl"
)
smH_PTJ0_Granular_xs = make_hgg_theory_pred_array(
    "theoryPred_smH_PTJ0_2018_ggHxH_NNLOPS.pkl", "theoryPred_smH_PTJ0_2018_ggHxH_NNLOPS_theoryUnc.pkl"
)
mjj_Hgg_xs = make_hgg_theory_pred_array(
    "theoryPred_mjj_2018_ggHxH_NNLOPS.pkl", "theoryPred_mjj_2018_ggHxH_NNLOPS_theoryUnc.pkl"
)
DEtajj_Hgg_xs = make_hgg_theory_pred_array(
    "theoryPred_Detajj_2018_ggHxH_NNLOPS.pkl", "theoryPred_Detajj_2018_ggHxH_NNLOPS_theoryUnc.pkl"
)
TauCJ_Hgg_xs = make_hgg_theory_pred_array(
    "theoryPred_TauCJ_2018_ggHxH_NNLOPS.pkl", "theoryPred_TauCJ_2018_ggHxH_NNLOPS_theoryUnc.pkl"
)

# no NNLOPS
smH_PTH_Hgg_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_Pt_2018_ggHxH.pkl", "theoryPred_Pt_2018_ggHxH_theoryUnc.pkl"
)
smH_PTH_EvenMoreMaximumGranularity_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_Pt_2018_ggHxH_EvenMoreGranular.pkl", "theoryPred_Pt_2018_ggHxH_EvenMoreGranular_theoryUnc.pkl"
)
Njets_Hgg_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_Njets_2018_ggHxH.pkl", "theoryPred_Njets_2018_ggHxH_theoryUnc.pkl"
)
yH_Granular_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_yH_2018_ggHxH_Granular.pkl", "theoryPred_yH_2018_ggHxH_Granular_theoryUnc.pkl"
)
smH_PTJ0_Hgg_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_smH_PTJ0_fineBinning_2018_ggHxH.pkl", "theoryPred_smH_PTJ0_fineBinning_2018_ggHxH_theoryUnc.pkl"
)
smH_PTJ0_Granular_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_smH_PTJ0_2018_ggHxH.pkl", "theoryPred_smH_PTJ0_2018_ggHxH_theoryUnc.pkl"
)
mjj_Hgg_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_mjj_2018_ggHxH.pkl", "theoryPred_mjj_2018_ggHxH_theoryUnc.pkl"
)
DEtajj_Hgg_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_Detajj_2018_ggHxH.pkl", "theoryPred_Detajj_2018_ggHxH_theoryUnc.pkl"
)
TauCJ_Hgg_xs_noNNLOPS = make_hgg_theory_pred_array(
    "theoryPred_TauCJ_2018_ggHxH.pkl", "theoryPred_TauCJ_2018_ggHxH_theoryUnc.pkl"
)

# powheg
smH_PTH_Hgg_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_Pt_2018_ggHxH_powheg.pkl", "theoryPred_Pt_2018_ggHxH_powheg_theoryUnc.pkl"
)
smH_PTH_EvenMoreMaximumGranularity_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_Pt_2018_ggHxH_EvenMoreGranular_powheg.pkl", "theoryPred_Pt_2018_ggHxH_EvenMoreGranular_powheg_theoryUnc.pkl"
)
Njets_Hgg_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_Njets_2018_ggHxH_powheg.pkl", "theoryPred_Njets_2018_ggHxH_powheg_theoryUnc.pkl"
)
yH_Granular_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_yH_2018_ggHxH_Granular_powheg.pkl", "theoryPred_yH_2018_ggHxH_Granular_powheg_theoryUnc.pkl"
)
smH_PTJ0_Hgg_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_smH_PTJ0_fineBinning_2018_ggHxH_powheg.pkl", "theoryPred_smH_PTJ0_fineBinning_2018_ggHxH_powheg_theoryUnc.pkl"
)
smH_PTJ0_Granular_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_smH_PTJ0_2018_ggHxH_powheg.pkl", "theoryPred_smH_PTJ0_2018_ggHxH_powheg_theoryUnc.pkl"
)
mjj_Hgg_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_mjj_2018_ggHxH_powheg.pkl", "theoryPred_mjj_2018_ggHxH_powheg_theoryUnc.pkl"
)
DEtajj_Hgg_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_Detajj_2018_ggHxH_powheg.pkl", "theoryPred_Detajj_2018_ggHxH_powheg_theoryUnc.pkl"
)
TauCJ_Hgg_xs_powheg = make_hgg_theory_pred_array(
    "theoryPred_TauCJ_2018_ggHxH_powheg.pkl", "theoryPred_TauCJ_2018_ggHxH_powheg_theoryUnc.pkl"
)


# TK stuff

TK_models = {
    "yukawa_coupdep": {"kappac": (-8, 8), "kappab": (-2, 2)},
    "yukawa_floatingBR": {"kappac": (-30, 30), "kappab": (-20, 20)},
    "top_coupdep_ctcg": {"ct": (0.0, 2.0), "cg": (-0.06, 0.08)},
    #"top_coupdep_ctcg": {"ct": (0.0, 3.5), "cg": (-0.2, 0.08)},
    "top_floatingBR_ctcg": {"ct": (-3, 3), "cg": (-0.15, 0.15)},
    "top_coupdep_ctcb": {"ct": (-1, 2), "cb": (-2.0, 2.0)},
    "top_floatingBR_ctcb": {"ct": (-3, 3), "cb": (-12, 12)},
}


"""
TK Legacy
"""


def unc_squared_per_mode(uncs, xs):
    return sum([(0.01 * unc * xs) ** 2 for unc in uncs])


# Uncertainties per mode, all in percentages of total XS
# first one is scale, second PDF, third alpha_s
uncs_VBF = [0.35, 2.1, 0.05]
uncs_WH = [0.6, 1.7, 0.9]
uncs_ZH = [3.4, 1.3, 0.9]
uncs_ttH = [7.5, 3.0, 2.0]
uncs_bbH = [22.0]
uncs_tH_t_ch = [10.6, 3.5, 1.2]
uncs_tH_s_ch = [2.1, 2.2, 0.2]
uncs_tH_W_associated = [5.8, 6.1, 1.5]
uncs_ggF = [5.65, 3.2]

# These seem to be taken from here https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV
YR4_ggF_n3lo = 4.852e01
YR4_VBF = 3.779e00
YR4_WH = 1.369e00
YR4_ZH = 8.824e-01
YR4_ttH = 5.065e-01
YR4_bbH = 4.863e-01
YR4_tH_t_ch = 7.426e-02
YR4_tH_s_ch = 2.875e-03
YR4_tH_W_associated = 0.000e00

YR4_totalXS = (
    YR4_ggF_n3lo
    + YR4_VBF
    + YR4_WH
    + YR4_ZH
    + YR4_ttH
    + YR4_bbH
    + YR4_tH_t_ch
    + YR4_tH_s_ch
    + YR4_tH_W_associated
)
YR4_xH = YR4_totalXS - YR4_ggF_n3lo
YR4_totalXS_uncertainty = 2.5

tot_unc_squared = 0.0
tot_unc_squared += unc_squared_per_mode(uncs_VBF, YR4_VBF)
tot_unc_squared += unc_squared_per_mode(uncs_WH, YR4_WH)
tot_unc_squared += unc_squared_per_mode(uncs_ZH, YR4_ZH)
tot_unc_squared += unc_squared_per_mode(uncs_ttH, YR4_ttH)
tot_unc_squared += unc_squared_per_mode(uncs_bbH, YR4_bbH)
tot_unc_squared += unc_squared_per_mode(uncs_tH_t_ch, YR4_tH_t_ch)
tot_unc_squared += unc_squared_per_mode(uncs_tH_s_ch, YR4_tH_s_ch)
tot_unc_squared += unc_squared_per_mode(uncs_tH_W_associated, YR4_tH_W_associated)
tot_unc_squared += unc_squared_per_mode(uncs_ggF, YR4_ggF_n3lo)

smH_unc_inclusive = np.sqrt(tot_unc_squared)
smH_unc_inclusive_fraction = smH_unc_inclusive / YR4_totalXS

# See https://github.com/maxgalli/differentialCombination2017/blob/master/LatestBinning.py#L187-L203
# This are the files taken from Vittorio
sm_uncs = {
    "njets": [
        0.03085762816223272,
        0.052913496610042327,
        0.08409612613286231,
        0.1204000132383638,
        0.09628940577730363,
    ]
}


def normalize(l, normalization=1.0):
    s = float(sum(l))
    return [e / s * normalization for e in l]


shape_njets = normalize([78.53302819, 30.8512632, 9.1331588, 2.41446376, 1.39813801])

add_quad = lambda *args: np.sqrt(sum([x ** 2 for x in args]))


def add_unc(shape_unc_perc):
    # Add in quadrature with incl xs uncertainty from YR4
    # Do this all in 'percent-space'
    return [add_quad(err, smH_unc_inclusive_fraction) for err in shape_unc_perc]


unc_fraction_njets = add_unc(sm_uncs["njets"])
