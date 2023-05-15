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
    matrix_bin_names,
    observable_specs,
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

        rfr_correlation = self.root_to_numpy_matrix(
            corr_matrix, list(self.rfr_coefficients_indices.values())
        )
        logger.debug(f"rfr_correlation before reordering: {rfr_correlation}")
        rfr_covariance = self.root_to_numpy_matrix(
            cov_matrix, list(self.rfr_coefficients_indices.values())
        )
        logger.debug(f"rfr_covariance before reordering: {rfr_covariance}")

        self.matrices["rfr_correlation"] = self.order_matrix(
            rfr_correlation, self.rfr_coefficients
        )
        self.matrices["rfr_covariance"] = self.order_matrix(
            rfr_covariance, self.rfr_coefficients
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
        hessian_correlation = self.root_histo_to_numpy_matrix(
            corr_hist, list(self.hessian_coefficients_indices.values())
        )
        logger.debug(f"hessian_correlation before reordering: {hessian_correlation}")

        cov_hist = f.Get("h_covariance")
        hessian_covariance = self.root_histo_to_numpy_matrix(
            cov_hist, list(self.hessian_coefficients_indices.values())
        )
        logger.debug(f"hessian_covariance before reordering: {hessian_covariance}")

        self.matrices["hessian_correlation"] = self.order_matrix(
            hessian_correlation, self.hessian_coefficients
        )
        self.matrices["hessian_covariance"] = self.order_matrix(
            hessian_covariance, self.hessian_coefficients
        )

        logger.debug(
            f"Found following coeffients and indices: {self.hessian_coefficients_indices}"
        )

        logger.debug(f"Found following matrices: {self.matrices}")

    def order_matrix(self, matrix, current_coefficients):
        """
        Order the matrix according to the order of the coefficients
        """
        ordered_matrix = []
        for i in self.coefficients:
            row = []
            for j in self.coefficients:
                row.append(
                    matrix[current_coefficients.index(i)][current_coefficients.index(j)]
                )
            ordered_matrix.append(row)

        return np.array(ordered_matrix)

    def dump(self, ouptut_dir, suffix="", observable=None):
        for matrix_name, matrix in self.matrices.items():
            logger.debug("First row of matrix: {}".format(matrix[0]))
            logger.debug("First row of ordered matrix: {}".format(matrix[0]))
            fig, ax = plt.subplots()
            number_size = 130 / len(self.coefficients)
            letter_size = (
                200 / len(self.coefficients) if len(self.coefficients) > 8 else 20
            )
            cmap = plt.get_cmap("bwr")
            cax = ax.matshow(
                matrix,
                cmap=cmap,
                vmin=-1 if "corr" in matrix_name else None,
                vmax=1 if "corr" in matrix_name else None,
            )
            cbar = plt.colorbar(cax, fraction=0.047, pad=0.01)
            cbar.ax.tick_params(labelsize=letter_size)

            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    c = matrix[j, i]
                    ax.text(
                        i,
                        j,
                        str("{:.4f}".format(c)),
                        va="center",
                        ha="center",
                        fontsize=number_size,
                    )

            try:
                labels = [SMEFT_parameters_labels[c] for c in self.coefficients]
            except KeyError:
                try:
                    labels = [TK_parameters_labels[c] for c in self.coefficients]
                except KeyError:
                    try:
                        labels = [matrix_bin_names[c] for c in self.coefficients]
                    except KeyError as e:
                        print(e)
                        labels = self.coefficients

            # set x axis tick labels to the bottom
            ax.set_xticks(np.arange(len(matrix)), minor=False)
            ax.set_yticks(np.arange(len(matrix)), minor=False)
            ax.set_xticklabels(labels, rotation=45, fontsize=letter_size)
            ax.set_yticklabels(labels, fontsize=letter_size)
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                labelbottom=True,
                top=False,
                labeltop=False,
            )
            ax.tick_params(axis="y", which="both", left=False, right=False)

            if observable is not None:
                ax.set_ylabel(observable_specs[observable]["x_plot_label"])

            # save
            hep.cms.label(loc=0, data=True, llabel="Internal", lumi=138, ax=ax)
            fig.tight_layout()
            fig.savefig(f"{ouptut_dir}/{matrix_name}{suffix}.png")
            fig.savefig(f"{ouptut_dir}/{matrix_name}{suffix}.pdf")
