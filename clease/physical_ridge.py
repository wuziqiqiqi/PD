from clease.regression import LinearRegression, Tikhonov
from clease.tools import split_dataset
from typing import List, Dict, Union, Callable
from clease import _logger
import numpy as np
from random import choice
import time


class PhysicalRidge(LinearRegression):
    """
    Physical Ridge is a special ridge regression scheme that enforces a
    convergent series. The physical motivation behind the choice of
    prior distributions is motivated by the fact that one expects that
    interactions strengths decays with both the number of atoms in the
    cluster and the diameter of the cluster. See for instance

    Cao, L., Li, C. and Mueller, T., 2018. The use of cluster expansions to
    predict the structures and properties of surfaces and nanostructured
    materials. Journal of chemical information and modeling, 58(12),
    pp.2401-2413.

    This fitting scheme uses Gaussian priors on the coefficients of the model

    P(M) = P_size(M)*P_dia(M), where

    P_size(M) = prod_i exp(-lamb_size*size_decay(size)*coeff_i^2)
    P_dia(M) = prod_i exp(-lamb_dia*dia_decay(dia)*coeff_i^2)

    where size_decay and dia_decay is a monotonically increasing function of
    the size and diameter respectively. The product goes over all coefficients
    in the model M.

    :param lamb_size: Prefactor in front of the size penalization

    :param lamb_dia: Prefactor in fron the the diameter penalization

    :param size_decay: The size_decay function in the priors explained above.
        It can be one of ['linear', 'exponential', 'polyN'],
        where N is any integer, or a callable function with
        the signature f(size), where size is the number of atoms in the
        cluster. If polyN is given the penalization is proportional to
        size**N

    :param dia_decay: The dia_decay function in the priors explained above.
        It can be one of ['linear', 'exponential', 'polyN']
        where N is any integer, of a callable function
        with the signature f(dia) where dia is the diameter.
        If polyN is given the penalization is proportional to dia**N

    :param normalize: If True the data will be normalized to unit variance
        and zero mean before fitting.
    """
    def __init__(self, lamb_size: float = 1e-6,
                 lamb_dia: float = 1e-6,
                 size_decay: Union[str, Callable[[int], float]] = 'linear',
                 dia_decay: Union[str, Callable[[int], float]] = 'linear',
                 normalize: bool = True) -> None:
        self.lamb_size = lamb_size
        self.lamb_dia = lamb_dia
        self._size_decay = get_size_decay(size_decay)
        self._dia_decay = get_dia_decay(dia_decay)
        self.sizes = []
        self.diameters = []
        self.normalize = normalize

    @property
    def size_decay(self) -> Callable[[int], float]:
        return self._size_decay

    @size_decay.setter
    def size_decay(self, decay: Union[str, Callable[[int], float]]) -> None:
        self._size_decay = get_size_decay(decay)

    @property
    def dia_decay(self) -> Callable[[int], float]:
        return self._dia_decay

    @dia_decay.setter
    def dia_decay(self, decay: Union[str, Callable[[int], float]]) -> None:
        self._dia_decay = get_dia_decay(decay)

    def sizes_from_names(self, names: List[str]) -> None:
        """
        Extract the sizes from a list of correlation function names

        :param names: List of cluster names.
            The length of the list has to match the
            number of columns in the X matrix passed to the fit method.
            Ex: ['c0', 'c1_1', 'c2_d0000_0_00']
        """
        self.sizes = [int(n[1]) for n in names]

    def diameters_from_names(self, names: List[str]) -> None:
        """
        Extract the diameters from a list of correltion function names

        :param names: List of cluster names.
            The length of the list has to match the
            number of columns in the X matrix passed to the fit method.
            Ex: ['c0', 'c1_1', 'c2_d0000_0_00']
        """
        diameters = []
        for n in names:
            if n[1] == '0' or n[1] == '1':
                diameters.append(0.0)
            else:
                dia_str = n.split('_')[1]
                dia = int(dia_str[1:])
                diameters.append(dia)
        self.diameters = diameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit ECIs

        :param X: Design matrix with correlation functions. The shape is N x M,
            where N is the number of data points and M is the number of
            correlation functions

        :param y: Vector with target values. The length of this vector is N
            (e.g. equal to the number of rows in X)
        """
        if X.shape[1] != len(self.sizes) or X.shape[1] != len(self.diameters):
            msg = "Inconsistent number of columns in X. "
            msg += f"Num. columns: {X.shape[1]} "
            msg += f"Num. sizes: {len(self.sizes)} "
            msg += f"Num. diameters: {len(self.diameters)}"
            raise ValueError(msg)

        start = 0
        if self.normalize:
            # Omit the bias term (first column since we are using normalized data)
            start = 1

        size_decay = np.array([self.size_decay(x) for x in self.sizes[start:]])
        dia_decay = np.array([self.dia_decay(x) for x in self.diameters[start:]])

        penalty = np.sqrt(self.lamb_size*size_decay + self.lamb_dia*dia_decay)
        regressor = Tikhonov(alpha=penalty, normalize=self.normalize)
        return regressor.fit(X, y)


def linear_size(size: int) -> float:
    if size == 0:
        return 0.0
    return size


def linear_dia(dia: int) -> float:
    return dia


def exponential_size(size: int) -> float:
    if size == 0:
        return 0.0
    return np.exp(size-1) - 1


def exponential_dia(dia: int) -> float:
    return np.exp(dia) - 1.0


def get_size_decay(decay: Union[str, Callable[[int], float]]
                   ) -> Callable[[int], float]:
    if isinstance(decay, str):
        if decay == 'linear':
            return linear_size
        elif decay == 'exponential':
            return exponential_size
        elif decay.startswith('poly'):
            power = int(decay[-1])
            return lambda size: size**power
        else:
            raise ValueError(f"Unknown decay type {decay}")
    elif callable(decay):
        return decay

    raise ValueError("size_decay has to be either a string or callable")


def get_dia_decay(decay: Union[str, Callable[[int], float]]
                  ) -> Callable[[int], float]:
    if isinstance(decay, str):
        if decay == 'linear':
            return linear_dia
        elif decay == 'exponential':
            return exponential_dia
        elif decay.startswith('poly'):
            power = int(decay[-1])
            return lambda dia: dia**power
        else:
            raise ValueError(f"Unknown decay type {decay}")
    elif callable(decay):
        return decay

    raise ValueError("dia_decay has to be either a string or callable")


def random_cv_hyper_opt(phys_ridge: PhysicalRidge,
                        params: Dict,
                        X: np.ndarray,
                        y: np.ndarray,
                        cv: int = 5,
                        num_trials: int = 100) -> Dict:
    """
    Estimate the hyper parameters of the Physical Ridge by random search.

    :param phys_ridge: Instance of the physical ridge class

    :param params: Dictionary with candiates for all parameters. Example:
        {
            'lamb_dia': [1e-12, 1e-11, 1e-4, 1e-3],
            'lamb_size': [1e-12, 1e-11, 1e-4, 1e-3],
            'size_decay': ['linear', 'exponential'],
            'dia_decay': ['linear', 'exponential']
        }
        on each iteration a random combination of the the specified
        candidates will be attempted.

    :param cv: Number of folds used for cross validation

    :param num_trials: Number of combinations of hyper parameters
        that will be tried
    """
    best_param = None
    best_cv = 0.0
    best_mse = 0.0
    best_coeff = None

    cv_params = []
    last_print = time.time()
    for i in range(num_trials):
        lamb_dia = choice(params.get('lamb_dia', [phys_ridge.lamb_dia]))
        lamb_size = choice(params.get('lamb_size', [phys_ridge.lamb_size]))
        size_decay = choice(params.get('size_decay', [phys_ridge.size_decay]))
        dia_decay = choice(params.get('dia_decay', [phys_ridge.dia_decay]))

        phys_ridge.lamb_dia = lamb_dia
        phys_ridge.lamb_size = lamb_size
        phys_ridge.size_decay = size_decay
        phys_ridge.dia_decay = dia_decay

        param_dict = {
            'lamb_dia': lamb_dia,
            'lamb_size': lamb_size,
            'size_decay': size_decay,
            'dia_decay': dia_decay,
        }

        partitions = split_dataset(X, y, nsplits=cv)

        cv_score = 0.0
        mse = 0.0
        for p in partitions:
            coeff = phys_ridge.fit(p['train_X'], p['train_y'])
            pred = p['validate_X'].dot(coeff)
            cv_score += np.sqrt(np.mean((pred - p['validate_y'])**2))

            pred = p['train_X'].dot(coeff)
            mse += np.sqrt(np.mean((pred - p['train_y'])**2))

        cv_score /= len(partitions)
        mse /= len(partitions)

        cv_params.append((cv_score, param_dict))
        if best_param is None or cv_score < best_cv:
            best_cv = cv_score
            best_mse = mse
            best_coeff = coeff
            best_param = param_dict

        if time.time() - last_print > 30:
            _logger(f"{i} of {num_trials}. CV: {best_cv*1000.0} meV/atom. "
                    f"MSE: {best_mse*1000.0} meV/atom. Params: {best_param}")
            last_print = time.time()

    cv_params = sorted(cv_params)
    res = {
        'best_coeffs': best_coeff,
        'best_params': best_param,
        'best_cv': best_cv,
        'cvs': [x[0] for x in cv_params],
        'params': [x[1] for x in cv_params]
    }
    return res
