from tkinter import font
import ROOT
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

import logging

logger = logging.getLogger(__name__)

from differential_combination_postprocess.cosmetics import (
    SMEFT_parameters_labels,
    TK_parameters_labels,
)


class MatricesExtractor:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.matrices = {}
        self.rfr_coefficients_indices = {}
        self.hessian_coefficients_indices = {}

    def root_to_numpy_matrix(self, root_matrix, indices):
        list_of_lists = []
        for i in indices:
            row = []
            for j in indices:
                row.append(root_matrix[i][j])
            list_of_lists.append(row)

        return np.array(list_of_lists)

    def rfr_to_numpy_corr_matrix(self, rfr, coefficients):
        list_of_lists = []
        for i in coefficients:
            row = []
            for j in coefficients:
                row.append(rfr.correlation(i, j))
            list_of_lists.append(row)

        return np.array(list_of_lists)

    def root_histo_to_numpy_matrix(self, root_hist, indices):
        list_of_lists = []
        for i in indices:
            row = []
            for j in indices:
                row.append(root_hist.GetBinContent(i, j))
            list_of_lists.append(row)

        return np.array(list_of_lists)

    def extract_from_roofitresult(self, root_file, roofit_result):
        """
        A ROOT file 'multidimfit...' is produced when combine is run with the option --saveFitResult.
        Inside here there should be pretty much everything we need, i.e.:
        - the covariance matrix
        - the correlation matrix
        - the correlation matrix in form of histogram, from which we extract the indices of the POIs
        """
        f = ROOT.TFile(root_file)
        rfr = f.Get(roofit_result)

        # first we get the indices of the coefficients from the 2D histogram
        # matrices do not seem to have named bins
        corr_hist = rfr.correlationHist()
        for i in range(1, corr_hist.GetNbinsX() + 1):
            label = corr_hist.GetXaxis().GetBinLabel(i)
            if label in self.coefficients:
                self.rfr_coefficients_indices[label] = i - 1
        self.rfr_coefficients = list(self.rfr_coefficients_indices.keys())

        logger.debug(
            f"Found following coeffients and indices: {self.rfr_coefficients_indices}"
        )

        corr_matrix = rfr.correlationMatrix()
        cov_matrix = rfr.covarianceMatrix()

        self.matrices["rfr_correlation"] = self.root_to_numpy_matrix(
            corr_matrix, list(self.rfr_coefficients_indices.values())
        )
        self.matrices["rfr_covariance"] = self.root_to_numpy_matrix(
            cov_matrix, list(self.rfr_coefficients_indices.values())
        )

        logger.debug(f"Found following matrices: {self.matrices}")

    def extract_from_robusthesse(self, root_file):
        """
        The options 
        --robustHesse 1 --robustFit 1
        seem to produce a ROOT file called robustHesse....root which contains h_covariance and h_correlation, TH2F objects
        """
        f = ROOT.TFile(root_file)

        corr_hist = f.Get("h_correlation")
        for i in range(1, corr_hist.GetNbinsX() + 1):
            label = corr_hist.GetXaxis().GetBinLabel(i)
            if label in self.coefficients:
                # note that this is different from the other one because when TH2Ds are treated as histos, the first bin has empty label; this does not happen with matrices
                self.hessian_coefficients_indices[label] = i
        self.hessian_coefficients = list(self.hessian_coefficients_indices.keys())
        self.matrices["hessian_correlation"] = self.root_histo_to_numpy_matrix(
            corr_hist, list(self.hessian_coefficients_indices.values())
        )

        cov_hist = f.Get("h_covariance")
        self.matrices["hessian_covariance"] = self.root_histo_to_numpy_matrix(
            cov_hist, list(self.hessian_coefficients_indices.values())
        )

        logger.debug(
            f"Found following coeffients and indices: {self.hessian_coefficients_indices}"
        )

        logger.debug(f"Found following matrices: {self.matrices}")

    def dump(self, ouptut_dir, suffix=""):
        for matrix_name, matrix in self.matrices.items():
            coefficients = (
                self.rfr_coefficients
                if "rfr_" in matrix_name
                else self.hessian_coefficients
            )
            fig, ax = plt.subplots()
            cmap = plt.get_cmap("bwr")
            cax = ax.matshow(
                matrix,
                cmap=cmap,
                vmin=-1 if "corr" in matrix_name else None,
                vmax=1 if "corr" in matrix_name else None,
            )
            cbar = plt.colorbar(cax)

            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    c = matrix[j, i]
                    ax.text(
                        i,
                        j,
                        str("{:.4f}".format(c)),
                        va="center",
                        ha="center",
                        fontsize=12,
                    )

            try:
                labels = [SMEFT_parameters_labels[c] for c in coefficients]
            except KeyError:
                try:
                    labels = [TK_parameters_labels[c] for c in coefficients]
                except KeyError:
                    labels = coefficients

            ax.set_xticks(np.arange(len(matrix)), minor=False)
            ax.set_yticks(np.arange(len(matrix)), minor=False)
            ax.set_xticklabels(labels, rotation=45, fontsize=12)
            ax.set_yticklabels(labels, rotation=45, fontsize=12)
            ax.tick_params(axis="x", which="both", bottom=False, top=False)
            ax.tick_params(axis="y", which="both", left=False, right=False)

            # save
            fig.savefig(f"{ouptut_dir}/{matrix_name}{suffix}.png")
            fig.savefig(f"{ouptut_dir}/{matrix_name}{suffix}.pdf")
