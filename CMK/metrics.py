import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment


class MetricCalculator:
    """Class for computing clustering metrics."""

    @staticmethod
    def accuracy_score(y_true, y_pred):
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
        return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

    @staticmethod
    def purity_score(y_true, y_pred):
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    @staticmethod
    def nmi_score(y_true, y_pred):
        return metrics.normalized_mutual_info_score(y_true, y_pred)

    @classmethod
    def evaluate(cls, y_true, y_pred):
        """Compute all metrics at once."""
        acc = cls.accuracy_score(y_true, y_pred)
        nmi = cls.nmi_score(y_true, y_pred)
        pur = cls.purity_score(y_true, y_pred)
        return acc, nmi, pur
