from clease import _logger
from clease import LinearRegression
from clease import Tikhonov
import numpy as np


class SequentialClusterRidge(LinearRegression):
    """
    SequentialClusterRidge is a fit method that optimizes the LOOCV over the
    regularization parameter as well as the cluster support. The method
    adds features in the design matrix X (see `fit` method) by including
    column by column. For each set of columns it performs a fit to a logspaced
    set of regularization parameters. The returned coefficients are the one
    from the model that has the smallest LOOCV.

    Parameters:

    alpha_min: float
        Minimum value of the regularization parameter alpha

    alpha_max: float
        Maximum value of the regularization parameter alpha

    num_alpha: int
        Number of alpha values
    """
    def __init__(self, min_alpha=1e-10, max_alpha=10.0, num_alpha=20):
        LinearRegression.__init__(self)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.num_alpha = num_alpha

    def _cv(self, X, coeff, y, l2scheme):
        """
        Calcualtes the cross validation score
        """
        pred = X.dot(coeff)
        prec = l2scheme.precision_matrix(X)
        dy = y - pred
        dy_loo = dy / (1 - np.diag(X.dot(prec).dot(X.T)))
        return np.sqrt(np.mean(dy_loo**2))

    def _print_summary(self, cvs, coeffs):
        """
        Prints a summary of the search
        """
        srt_idx = np.argsort(cvs)
        _logger("--------------------------------------------")
        _logger("       SUPPORT EXPANDING L2 SUMMARY         ")
        _logger("--------------------------------------------")
        for i in range(20):
            _logger("Num. coeff: {:9d} CV: {:9.3f} meV/atom"
                    "".format(len(coeffs[srt_idx[i]]), cvs[srt_idx[i]]))
        _logger("--------------------------------------------")

    def fit(self, X, y):
        """
        Performs the fitting

        Parameters:

        X: np.ndarray
            Design matrix of size (N x M). During the CV optimization columns
            of X will be added one by one starting with a model consisting
            of the two first columns.

        y: np.ndarray
            Vector of length N
        """
        numFeat = X.shape[1]
        alphas = np.logspace(
            np.log10(self.min_alpha), np.log10(self.max_alpha),
            self.num_alpha)

        coeffs = []
        cvs = []
        for i in range(2, numFeat):
            for j in range(len(alphas)):
                scheme = Tikhonov(alpha=alphas[j])
                design = X[:, :i]
                coeff = scheme.fit(design, y)
                cv = self._cv(design, coeff, y, scheme)
                cvs.append(cv)
                coeffs.append(coeff)

        best_cv = np.argmin(cvs)
        res = np.zeros(numFeat)
        best_coeff = coeffs[best_cv]
        res[:len(best_coeff)] = best_coeff
        self._print_summary(cvs, coeffs)
        return res
