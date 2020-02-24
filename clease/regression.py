"""Collection of classess to perform regression."""
import numpy as np
from numpy.linalg import pinv
from clease.dataNormalizer import DataNormalizer


class LinearRegression(object):
    def __init__(self):
        self._weight_matrix = None
        self.tol = 1E-8

    @property
    def weight_matrix(self):
        return self._weight_matrix

    @weight_matrix.setter
    def weight_matrix(self, matrix):
        self._weight_matrix = matrix

    def _ensure_weight_matrix_consistency(self, data):
        """Raise an error if the dimensions of the
           weight matrix is not consistent.

        Parameters:

        data: numpy.ndarray
            y-values in the fit
        """
        if self._weight_matrix is not None:
            if self._weight_matrix.shape[1] != len(data):
                raise ValueError(f"The provided weight matrix needs to have "
                                 f"dimension {len(data)}x{len(data)}, "
                                 f"{self._weight_matrix.shape} given")

    def fit(self, X, y):
        """Fit a linear model by performing ordinary least squares

        y = Xc

        Parameters:

        X: Design matrix (NxM)
        y: Data points (vector of length N)
        """
        self._ensure_weight_matrix_consistency(y)

        # We use SVD to carry out the fit
        U, D, V_h = np.linalg.svd(X, full_matrices=False)
        V = V_h.T
        diag_item = np.zeros_like(D)
        mask = np.abs(D) > self.tol
        diag_item[mask] = 1.0/D[mask]
        coeff = V.dot(np.diag(diag_item)).dot(U.T).dot(y)
        return coeff

    def precision_matrix(self, X):
        U, D, V_h = np.linalg.svd(X, full_matrices=False)
        V = V_h.T
        diag = np.zeros_like(D)
        mask = np.abs(D) > self.tol
        diag[mask] = 1.0/D[mask]**2
        return V.dot(np.diag(diag)).dot(V.T)

    @staticmethod
    def get_instance_array():
        return [LinearRegression()]

    def is_scalar(self):
        return False

    def get_scalar_parameter(self):
        raise ValueError("Fitting scheme is not described by a scalar "
                         "parameter!")

    @property
    def support_fast_loocv(self):
        return True


class Tikhonov(LinearRegression):
    """Ridge regularization.

    Parameters:

    alpha: float, 1D or 2D numpy array
        regularization term
        - float: A single regularization coefficient is used for all features.
                 Tikhonov matrix is T = alpha * I (I = identity matrix).
        - 1D array: Regularization coefficient is defined for each feature.
                    Tikhonov matrix is T = diag(alpha) (the alpha values are
                    put on the diagonal).
                    The length of array should match the number of features.
        - 2D array: Full Tikhonov matrix supplied by a user.
                    The dimensions of the matrix should be M * M where M is the
                    number of features.

    normalize: bool
        If True each feature will be normalized to before fitting
    """

    def __init__(self, alpha=1E-5, penalize_bias_term=False, normalize=True):
        LinearRegression.__init__(self)
        self.alpha = alpha
        self.penalize_bias_term = penalize_bias_term
        self.normalize = normalize

    def _get_tikhonov_matrix(self, num_clusters):
        if isinstance(self.alpha, np.ndarray):
            if len(self.alpha.shape) == 1:
                tikhonov = np.diag(self.alpha)
            elif len(self.alpha.shape) == 2:
                tikhonov = self.alpha
            else:
                raise ValueError("Matrix have to have dimension 1 or 2")
        else:
            # Alpha is a floating point number
            tikhonov = np.identity(num_clusters)
            if not self.penalize_bias_term:
                tikhonov[0, 0] = 0.0
            tikhonov *= np.sqrt(self.alpha)
        return tikhonov

    def fit(self, X, y):
        """Fit coefficients based on Ridge regularizeation."""
        self._ensure_weight_matrix_consistency(y)

        if self.weight_matrix is None:
            W = np.ones(len(y))
        else:
            W = np.diag(self.weight_matrix)

        X_fit = X
        y_fit = y
        if self.normalize:
            if np.any(np.abs(X[:, 0] - 1.0) > 1e-16):
                msg = "Tikhonov: Expect that the first column in X corresponds"
                msg += "to a bias term. Therefore, all entries should be 1."
                msg += f"Got:\n{X[:, 0]}\n"
                raise ValueError(msg)
            normalizer = DataNormalizer()
            X_fit, y_fit = normalizer.normalize(X[:, 1:], y)

        precision = self.precision_matrix(X_fit)
        coeff = precision.dot(X_fit.T.dot(W*y_fit))

        if self.normalize:
            coeff_with_bias = np.zeros(len(coeff)+1)
            coeff_with_bias[1:] = normalizer.convert(coeff)
            coeff_with_bias[0] = normalizer.bias(coeff)
            coeff = coeff_with_bias
        return coeff

    def precision_matrix(self, X):
        """Calculate the presicion matrix."""
        num_features = X.shape[1]
        tikhonov = self._get_tikhonov_matrix(num_features)

        if tikhonov.shape != (num_features, num_features):
            raise ValueError("The dimensions of Tikhonov matrix do not match "
                             "the number of clusters!")

        W = self.weight_matrix
        if W is None:
            W = np.eye(X.shape[0])
        precision = pinv(X.T.dot(W.dot(X)) + tikhonov.T.dot(tikhonov))
        return precision

    @staticmethod
    def get_instance_array(alpha_min, alpha_max, num_alpha=10, scale='log'):
        if scale == 'log':
            alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                                int(num_alpha), endpoint=True)
        else:
            alpha = np.linspace(alpha_min, alpha_max, int(num_alpha),
                                endpoint=True)
        return [Tikhonov(alpha=a) for a in alpha]

    def is_scalar(self):
        return isinstance(self.alpha, float)

    def get_scalar_parameter(self):
        if self.is_scalar():
            return self.alpha
        LinearRegression.get_scalar_parameter(self)


class Lasso(LinearRegression):
    """LASSO regularization.

    Parameter:

    alpha: float
        regularization coefficient
    """

    def __init__(self, alpha=1E-5):
        LinearRegression.__init__(self)
        self.alpha = alpha

    def fit(self, X, y):
        """Fit coefficients based on LASSO regularizeation."""
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=self.alpha, fit_intercept=False, copy_X=True,
                      normalize=True, max_iter=1e6)
        lasso.fit(X, y)
        return lasso.coef_

    @property
    def weight_matrix(self):
        return LinearRegression.weight_matrix(self)

    @weight_matrix.setter
    def weight_matrix(self, X):
        raise NotImplementedError("Currently Lasso does not support "
                                  "data weighting.")

    @staticmethod
    def get_instance_array(alpha_min, alpha_max, num_alpha=10, scale='log'):
        if scale == 'log':
            alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                                int(num_alpha), endpoint=True)
        else:
            alpha = np.linspace(alpha_min, alpha_max, int(num_alpha),
                                endpoint=True)
        return [Lasso(alpha=a) for a in alpha]

    def is_scalar(self):
        return True

    def get_scalar_parameter(self):
        return self.alpha

    def precision_matrix(self, X):
        raise NotImplementedError("Precision matrix for LASSO is not "
                                  "implemented.")

    @property
    def support_fast_loocv(self):
        return False
