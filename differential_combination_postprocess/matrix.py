from tkinter import font
import ROOT
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

import logging

logger = logging.getLogger(__name__)


class MatricesExtractor:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.matrices = {}
        self.coefficients_indices = {}

    def root_to_numpy_matrix(self, root_matrix, indices):
        list_of_lists = []
        for i in indices:
            row = []
            for j in indices:
                row.append(root_matrix[i][j])
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
                self.coefficients_indices[label] = i - 1
        self.coefficients = list(self.coefficients_indices.keys())

        logger.debug(
            f"Found following coeffients and indices: {self.coefficients_indices}"
        )

        corr_matrix = rfr.correlationMatrix()
        cov_matrix = rfr.covarianceMatrix()

        self.matrices["rfr_correlation"] = self.root_to_numpy_matrix(
            corr_matrix, list(self.coefficients_indices.values())
        )
        self.matrices["rfr_covariance"] = self.root_to_numpy_matrix(
            cov_matrix, list(self.coefficients_indices.values())
        )

        logger.debug(f"Found following matrices: {self.matrices}")

    def extract_from_robusthesse(self, root_file, hessian):
        """
        This has to be called AFTER extract_from_roofitresult since we need the indices of the coefficients.

        see https://github.com/cms-analysis/CombineHarvester/blob/653c523ebe6e67e7a8214366bfe6bd99f7d87d99/CombineTools/python/combine/TaylorExpand.py#L584-L593
        """
        f = ROOT.TFile(root_file)
        hessian = f.Get(hessian)

        cov_matrix = hessian.Clone()
        self.matrices["hessian_covariance"] = self.root_to_numpy_matrix(
            cov_matrix, list(self.coefficients_indices.values())
        )

        # cov_matrix.Invert()
        corr_matrix = cov_matrix.Clone()
        for i in range(corr_matrix.GetNrows()):
            for j in range(corr_matrix.GetNcols()):
                try:
                    corr_matrix[i][j] = cov_matrix[i][j] / sqrt(
                        cov_matrix[i][i] * cov_matrix[j][j]
                    )
                except ValueError:
                    print(f"cov_matrix[{i}][{j}] = {cov_matrix[i][j]}")
                    print(f"cov_matrix[{i}][{i}] = {cov_matrix[i][i]}")
                    print(f"cov_matrix[{j}][{j}] = {cov_matrix[j][j]}\n")
                    corr_matrix[i][j] = 0.0
        self.matrices["hessian_correlation"] = self.root_to_numpy_matrix(
            corr_matrix, list(self.coefficients_indices.values())
        )

    def dump(self, ouptut_dir):
        for matrix_name, matrix in self.matrices.items():
            fig, ax = plt.subplots()
            cmap = plt.get_cmap("Wistia")
            ax.matshow(matrix, cmap=cmap)

            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    c = matrix[j, i]
                    ax.text(
                        i,
                        j,
                        str("{:.4f}".format(c)),
                        va="center",
                        ha="center",
                        fontsize=10,
                    )

            ax.set_xticks(np.arange(len(matrix)), minor=False)
            ax.set_yticks(np.arange(len(matrix)), minor=False)
            ax.set_xticklabels(self.coefficients, rotation=45, fontsize=10)
            ax.set_yticklabels(self.coefficients, rotation=45, fontsize=10)
            ax.tick_params(axis="x", which="both", bottom=False, top=False)
            ax.tick_params(axis="y", which="both", left=False, right=False)

            # save
            fig.savefig(f"{ouptut_dir}/{matrix_name}.png")
            fig.savefig(f"{ouptut_dir}/{matrix_name}.pdf")
