#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random Effects SEM."""
import pandas as pd
import numpy as np
from .model_means import ModelMeans
from .utils import chol_inv, chol_inv2, cov, calc_zkz, delete_mx
from .univariate_blup import blup
from scipy.linalg import solve_sylvester
from collections import defaultdict
from .solver import Solver
from itertools import combinations


class ModelEffects(ModelMeans):
    """
    Random Effects model.

    Random Effects SEM can be interpreted as a generalization of Linear Mixed
    Models (LMM) to SEM.
    """

    matrices_names = tuple(list(ModelMeans.matrices_names) + ['d'])
    symb_rf_covariance = '~R~'

    def __init__(self, description: str, mimic_lavaan=False, baseline=False,
                 cov_diag=False, intercepts=True, d_mode='diag', effects=None):
        """
        Instantiate Random Effects SEM.

        Parameters
        ----------
        description : str
            Model description in semopy syntax.
        mimic_lavaan: bool, optional
            If True, output variables are correlated and not conceptually
            identical to indicators. lavaan treats them that way, but it's
            less computationally effective. The default is False.
        baseline : bool, optional
            If True, the model will be set to baseline model.
            Baseline model here is an independence model where all variables
            are considered to be independent with zero covariance. Only
            variances are estimated. The default is False.
        cov_diag : bool, optional
            If cov_diag is True, then there are no covariances parametrised
            unless explicitly specified. The default is False.
        intercepts: bool, optional
            If True, intercepts are also modeled. Intercept terms can be
            accessed via "1" symbol in a regression equation, i.e. x1 ~ 1. The
            default is False.
        d_mode : str, optional
            Mode of D matrix. If "diag", then D has unique params on the
            diagonal. If "full", then D is fully parametrised. If
            "scale", then D is an identity matrix, multiplied by a single
            variance parameter (scalar). Utilised only if effect names
            are not provided. The default is "diag".
        effects : list, optional
            List of effects name. Must correspond to cetrain columns in data.
            If None and effects are not provided in syntax, then matrix D
            is parametrised in accordance to d_mode. The default is None.
        Returns
        -------
        None.

        """
        self.dict_effects[self.symb_rf_covariance] = self.effect_rf_covariance
        self.d_mode = d_mode
        if effects is None:
            effects = set()
        if len(effects) > 1:
            raise Exception("ModelEffects supports only one random effect."
                            " Consider using ModelGeneralizedEffects.")
        self.effects_names = effects
        self.effects_loadings = defaultdict(float)
        super().__init__(description, mimic_lavaan=mimic_lavaan, 
                         cov_diag=cov_diag, baseline=baseline,
                         intercepts=intercepts)

        self.objectives = {'REML': (self.obj_reml, self.grad_reml),
                           'REML2': (self.obj_reml2, self.grad_reml2),
                           'ML': (self.obj_matnorm, self.grad_matnorm)}

    def before_classification(self, effects: dict, operations: dict):
        """
        Preprocess effects and operations if necessary before classification.

        Parameters
        ----------
        effects : dict
            Dict returned from parse_desc.

        operations: dict
            Dict of operations as returned from parse_desc.

        Returns
        -------
        None.

        """
        super().before_classification(effects, operations)
        symb = self.symb_rf_covariance
        regr = self.symb_regression
        eff_names = self.effects_names
        loadings = self.effects_loadings
        eff_regr = effects[regr]
        eff_rf = effects[symb]
        for v, rvs in eff_regr.items():
            to_rem = list()
            for rv in rvs:
                if rv in eff_names:
                    eff_rf[v][v] = None
                    loadings[v] = 0.05
                    to_rem.append(rv)
                    continue
            for rv in to_rem:
                del rvs[rv]

    def preprocess_effects(self, effects: dict):
        """
        Run a routine just before effects are applied.

        Used to apply random effect variance
        Parameters
        -------
        effects : dict
            Mapping opcode->lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        super().preprocess_effects(effects)
        mode = self.d_mode
        symb = self.symb_rf_covariance
        obs = self.vars['observed']
        eff_names = self.effects_names
        loadings = self.effects_loadings
        if not eff_names:
            if mode in ('diag', 'full'):
                for v in obs:
                    t = effects[symb][v]
                    if v not in t:
                        t[v] = None
                        loadings[v] = list()
                if mode == 'full':
                    for a, b in combinations(obs, 2):
                        t = effects[symb][a]
                        tt = effects[symb][b]
                        if (v not in t) and (v not in tt):
                            t[b] = None
            else:
                if mode != 'scale':
                    raise Exception(f'Unknown mode "{mode}".')
                param = 'paramD'
                for v in obs:
                    t = effects[symb][v][v] = param
                    loadings[v] = list()

    def build_d(self):
        """
        D matrix is a covariance matrix for random effects across columns.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        names = self.vars['observed']
        n = len(names)
        mx = np.zeros((n, n))
        return mx, (names, names)

    def load(self, data, group=None, k=None, cov=None, clean_slate=False,
             n_samples=None, obj='ML'):
        """
        Load dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Data with columns as variables.
        group : str, optional
            Name of column with group labels. If not provided, predefined
            columns are used.
        k : pd.DataFrame, optional
            Covariance matrix across rows, i.e. kinship matrix. If None,
            identity is assumed. The default is None.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. Used only for variance
            starting values. The default is None.
        clean_slate : bool, optional
            If True, resets parameters vector. The default is False.
        n_samples : int, optional
            Redunant for ModelEffects. The default is None.
        obj : str, optional
            Objective fuction name necessary to do the initial data
            preparation. The default is 'ML'.

        KeyError
            Rises when there are missing variables from the data.
        Exception
            Rises when group parameter is None.
        Returns
        -------
        None.

        """
        if data is None:
            if not hasattr(self, 'mx_data'):
                raise Exception("Data must be provided.")
            if clean_slate:
                self.prepare_params()
            return
        else:
            data = data.copy()
        obs = self.vars['observed']
        exo = self.vars['observed_exogenous']
        if self.intercepts:
            data['1'] = 1.0
        cols = data.columns
        missing = (set(obs) | set(exo)) - set(set(cols))
        if missing:
            t = ', '.join(missing)
            raise KeyError('Variables {} are missing from data.'.format(t))
        self.load_data(data, k=k, covariance=cov, group=group)
        if obj == 'REML':
            if self.__loaded != 'REML':
                self.load_reml()
        elif obj == 'ML':
            if self.__loaded != 'ML':
                self.load_ml()
        self.load_starting_values()
        if clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()

    def _fit(self, obj='REML', solver='SLSQP', **kwargs):
        fun, grad = self.get_objective(obj)
        solver = Solver(solver, fun, grad, self.param_vals,
                        constrs=self.constraints,
                        bounds=self.get_bounds(),
                        **kwargs)
        res = solver.solve()
        res.name_obj = obj
        self.param_vals = res.x
        self.update_matrices(res.x)
        self.last_result = res
        return res

    def fit(self, data=None, group=None, k=None, cov=None, obj='ML',
            solver='SLSQP', clean_slate=False, regularization=None, **kwargs):
        """
        Fit model to data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data with columns as variables. The default is None.
        group : str, optioal
            Name of column in data with group labels. Overrides effect names
            provided by the constructor. The default is None.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. The default is None.
        obj : str, optional
            Objective function to minimize. Possible values are 'REML', 'ML'.
            The default is 'REML'.
        solver : TYPE, optional
            Optimizaiton method. Currently scipy-only methods are available.
            The default is 'SLSQP'.
        clean_slate : bool, optional
            If False, successive fits will be performed with previous results
            as starting values. If True, parameter vector is reset each time
            prior to optimization. The default is False.
        regularization
            Special structure as returend by create_regularization function.
            If not None, then a regularization will be applied to a certain
            parameters in the model. The default is None.

        Raises
        ------
        Exception
            Rises when attempting to use MatNorm in absence of full data.

        Returns
        -------
        SolverResult
            Information on optimization process.

        """
        self.load(data=data, cov=cov, group=group, k=k,
                  clean_slate=clean_slate, obj=obj)
        if not hasattr(self, 'mx_data'):
            raise Exception('Full data must be supplied.')
        if obj == 'REML':
            # if self.__loaded != 'REML':
            #     self.load_reml()
            res_reml = self._fit(obj='REML', solver=solver, **kwargs)
            self.load_ml(fake=True)
            sigma, (self.mx_m, _) = self.calc_sigma()
            self.mx_r_inv = chol_inv(self.calc_l(sigma))
            self.mx_w_inv = self.calc_t_inv(sigma)[0]
            res_reml2 = self._fit(obj='REML2', solver=solver, **kwargs)
            return (res_reml, res_reml2)
        elif obj == 'ML':
            # if self.__loaded != 'ML':
            #     self.load_ml()
            res = self._fit(obj='ML', solver=solver, **kwargs)
            return res
        else:
            raise NotImplementedError(f'Unknown objective {obj}.')      

    def effect_rf_covariance(self, items: dict):
        """
        Work through random effects covariance operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        mx = self.mx_d
        rows, cols = self.names_d
        for lv, rvs in items.items():
            for rv, mult in rvs.items():
                name = None
                try:
                    val = float(mult)
                    active = False
                except (TypeError, ValueError):
                    active = True
                    if mult is not None:
                        if mult != self.symb_starting_values:
                            name = mult
                        else:
                            active = False
                    val = None
                if name is None:
                    self.n_param_cov += 1
                    name = '_c%s' % self.n_param_cov
                i, j = rows.index(lv), cols.index(rv)
                ind = (i, j)
                if i == j:
                    bound = (0, None)
                    symm = False
                else:
                    if self.baseline:
                        continue
                    bound = (None, None)
                    symm = True
                self.add_param(name, matrix=mx, indices=ind, start=val,
                               active=active, symmetric=symm, bound=bound)

    def set_fim_means(self):
        """
        Substitute true FIM matrix with means-only FIM matrix.

        A trick to reduce GWAS time as only mean components are subject to
        analysis.
        Returns
        -------
        None.

        """
        
        self.calc_fim = self.calc_fim_means

    # def load_starting_values(self):
    #     """
    #     Load starting values for parameters from empirical data.

    #     Returns
    #     -------
    #     None.

    #     """
    #     trans_data = self.mx_data_transformed.copy()
    #     obs = self.vars['observed']
    #     loadings = self.effects_loadings
    #     k = self.mx_s
    #     for v in loadings:
    #         i = obs.index(v)
    #         y = trans_data[i]
    #         up, s = blup(y - y.mean(), k)
    #         if s[0] > 0.01 and s[1] > 0.01:
    #             trans_data[i] -= up
    #             loadings[v] = s[1]
    #     cov = self.mx_cov.copy()
    #     self.mx_cov = np.cov(trans_data)
    #     if len(self.mx_cov.shape) < 2:
    #         self.mx_cov = self.mx_cov.reshape((1, 1))
    #     super().load_starting_values()
    #     self.mx_cov = cov


    '''
    ----------------------------LINEAR ALGEBRA PART---------------------------
    ----------------------The code below is responsible-----------------------
    ------------------for covariance structure computations-------------------
    '''

    '''
    ---------------------------R and W matrices-------------------------------
    '''

    def calc_l(self, sigma: np.ndarray):
        """
        Calculate covariance across columns matrix R.

        Parameters
        ----------
        sigma : np.ndarray
            Sigma matrix.

        Returns
        -------
        tuple
            R matrix.

        """
        n = self.num_n
        return n * sigma + self.mx_d * self.trace_zkz

    def calc_l_grad(self, sigma_grad: list):
        """
        Calculate gradient of R matrix.

        Parameters
        ----------
        sigma_grad : list
            Sigma gradient values.

        Returns
        -------
        grad : list
            Gradient of R matrix.

        """
        grad = list()
        n = self.num_n
        for g, df in zip(sigma_grad, self.mx_diffs):
            g = n * g
            if df[6] is not None:  # D
                g += df[6] * self.trace_zkz
            grad.append(g)
        return grad

    def calc_t_inv(self, sigma: np.ndarray):
        """
        Calculate inverse and logdet of covariance across rows matrix W.

        This function estimates only inverse of W. There was no need in package
        to estimate W.
        Parameters
        ----------
        sigma : np.ndarray
            Sigma matrix.

        Returns
        -------
        tuple
        R^{-1} and ln|R|.

        """
        w = self.calc_t(sigma)
        if np.any(w < 1e-9):
            raise np.linalg.LinAlgError
        return 1 / w, np.sum(np.log(w))

    def calc_t(self, sigma: np.ndarray):
        """
        Calculate W matrix.

        Parameters
        ----------
        sigma : np.ndarray
            Sigma matrix.

        Returns
        -------
        np.ndarray
            W matrix.

        """
        tr_sigma = np.trace(sigma)
        tr_d = np.trace(self.mx_d)
        return tr_d * self.mx_s + tr_sigma

    def calc_t_grad(self, sigma_grad: list):
        """
        Calculate gradient of W matrix.

        Parameters
        ----------
        sigma_grad : list
            Gradient of Sigma matrix.

        Returns
        -------
        grad : list
            Gradient of W.

        """
        grad = list()
        for g, df in zip(sigma_grad, self.mx_diffs):
            if len(g.shape):
                g = np.trace(g) * self.mx_i_n
            if df[6] is not None:  # D
                g += np.trace(df[6]) * self.mx_s
            grad.append(g)
        return grad

    def calc_t_inv_grad(self, inv_w: np.ndarray, sigma_grad: list):
        """
        Calculate gradient of W inverse and logdet matrix.

        Parameters
        ----------
        inv_w : np.ndarray
            Inverse of W matrix.
        sigma_grad : list
            Gradient of Sigma matrix.

        Returns
        -------
        grad : list
            Gradient of inverse of W.
        grad_logdet : list
            Gradient of logdet of W.

        """
        grad, grad_logdet = list(), list()
        inv_w_t = inv_w * self.mx_s
        inv_w_d = -(inv_w ** 2)
        inv_w_d_t = inv_w_d * self.mx_s
        for g, df in zip(sigma_grad, self.mx_diffs):
            gw = g
            gl = g
            if len(g.shape):
                tr = np.trace(g)
                gw = tr * inv_w_d
                gl = tr * inv_w
            if df[6] is not None:  # D
                tr = np.trace(df[6])
                gw += tr * inv_w_d_t
                gl += tr * inv_w_t
            grad.append(gw)
            grad_logdet.append(np.sum(gl))
        return grad, grad_logdet

    '''
    ---------------------Preparing structures for a more-----------------------
    ------------------------efficient computations-----------------------------
    '''

    def load_data(self, data: pd.DataFrame, group=None, k=None,
                  covariance=None):
        """
        Load dataset from data matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with columns as variables and rows as observations.
        group : str, optional
            Name of column that correspond to group labels. If not provided,
            items from effects_names are used. The default is None.
        k : pd.DataFrame or tuple
            Covariance matrix betwen groups. If None, then it's assumed to be
            an identity matrix. Alternatively, a tuple of (ZKZ^T, S, Q) can be
            provided where ZKZ^T = Q S Q^T an eigendecomposition of ZKZ^T. S
            must be provided in the vector/list form. The default is None.
        covariance : pd.DataFrame, optional 
            Custom covariance matrix. The default is None.

        Returns
        -------
        None.

        """
        obs = self.vars['observed']
        if type(k) in (tuple, list):
            if len(k) != 3:
                raise Exception("Both ZKZ^T and its eigendecomposition must "
                                "be provided.")
        if group is None:
            group = next(iter(self.effects_names))
        self.mx_g_orig = data[self.vars['observed_exogenous']].values.T
        if len(self.mx_g_orig.shape) != 2:
            self.mx_g_orig = self.mx_g_orig[np.newaxis, :]
        self.mx_g = self.mx_g_orig
        self.mx_data = data[obs].values
        self.n_samples, self.n_obs = self.mx_data.shape
        self.num_m = len(set(self.vars['observed']) - self.vars['latent'])
        self.passed_k = k
        if type(k) is tuple:
            self.mx_zkz, self.mx_sk, self.mx_q = k
            self._ktuple = True
        else:
            self._ktuple = False
            self.mx_zkz = calc_zkz(data[group], k)
        self.__loaded = None
        self.load_cov(covariance[obs].loc[obs]
                      if covariance is not None else cov(self.mx_data))

    def load_ml(self, fake=False):
        self.trace_zkz = np.trace(self.mx_zkz)
        if self._ktuple:
            self.mx_s = self.mx_sk
            q = self.mx_q
        else:
            s, q = np.linalg.eigh(self.mx_zkz)
            self.mx_s, self.mx_q = s, q
        self.mx_data_transformed = self.mx_data.T @ q
        self.mx_g = self.mx_g_orig @ q
        self.num_n = self.mx_data_transformed.shape[1]
        self.mx_i_n = np.ones(self.num_n)
        self.__loaded = 'ML'

    def load_reml(self):
        g = self.mx_g_orig
        try:
            s = np.identity(g.shape[1]) - g.T @ chol_inv(g @ g.T) @ g
        except ValueError:
            raise Exception("REML should not be used when there are no"
                            " either intercepts or exogenous variables in "
                            "Gamma matrices.")
        d, q = np.linalg.eigh(s)
        rank_dec = 0
        for i in d:
            if abs(i) < 1e-8:
                rank_dec += 1
            else:
                break
        d = np.diag(d)[rank_dec:, :]
        a = d @ q.T
        azkza = a @ self.mx_zkz @ a.T
        self.trace_zkz = np.trace(azkza)
        s, q = np.linalg.eigh(azkza)
        self.mx_s = s
        self.mx_data_transformed = self.mx_data.T @ a.T @ q
        self.num_n = self.mx_data_transformed.shape[1]
        self.mx_i_n = np.ones(self.num_n)
        self.__loaded = 'REML'

    '''
    ---------------Matrix Variate Normal Restricted Maximum Likelihood---------
    '''

    def obj_reml(self, x: np.ndarray):
        """
        Restricted loglikelihood of matrix-variate normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood.

        """
        self.update_matrices(x)
        sigma, _ = self.calc_sigma()
        try:
            r = self.calc_l(sigma)
            r_inv, logdet_r = chol_inv2(r)
            w_inv, logdet_w = self.calc_t_inv(sigma)
        except np.linalg.LinAlgError:
            return np.inf
        mx = self.mx_data_transformed
        tr_r = np.trace(r)
        n, m = self.num_n, self.num_m
        r_center = r_inv @ mx
        center_w = mx * w_inv
        tr = tr_r * np.einsum('ji,ji->', center_w, r_center)
        return tr + m * logdet_w + n * logdet_r - n * m * np.log(tr_r)

    def grad_reml(self, x: np.ndarray):
        """
        Gradient of REML objective of matrix-variate normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of REML objective.

        """
        self.update_matrices(x)
        grad = np.zeros_like(x)
        sigma, (m, c) = self.calc_sigma()
        try:
            r = self.calc_l(sigma)
            r_inv = chol_inv(r)
            w_inv, _ = self.calc_t_inv(sigma)
        except np.linalg.LinAlgError:
            grad[:] = np.inf
            return grad
        center = self.mx_data_transformed
        A = r_inv @ center
        B = (center * w_inv).T
        tr_ab = np.einsum('ij,ji->', A, B)
        tr_r = np.trace(r)
        V1 = center.T @ A
        V3 = B @ r_inv
        V2 = self.num_n * r_inv / tr_r - A @ V3
        sigma_grad = self.calc_sigma_grad(m, c)
        r_grad = self.calc_l_grad(sigma_grad)
        w_grad, w_grad_logdet = self.calc_t_inv_grad(w_inv, sigma_grad)
        n, m = self.num_n, self.num_m
        for i, (d_r, d_w, d_l) in enumerate(zip(r_grad, w_grad,
                                                w_grad_logdet)):
            g = 0.0
            tr_long = 0.0
            if len(d_r.shape):
                tr_long += np.einsum('ij,ji->', V2, d_r)
                tr_dr = np.trace(d_r)
                g += tr_dr * tr_ab
                g -= m * n * tr_dr / tr_r
            if len(d_w.shape):
                tr_long += np.einsum('ii,i->', V1, d_w)
                g += m * d_l
            g += tr_r * tr_long
            grad[i] = g
        return grad

    '''
    ------------------Matrix Variate REML (II-nd stage)-----------------------
    '''

    def obj_reml2(self, x: np.ndarray):
        """
        Loglikelihood of matrix-variate normal distribution given Sigma.

        For a second stage of REML estimation.
        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood.

        """
        self.update_matrices(x)
        mean = self.calc_mean(self.mx_m)
        center = self.mx_data_transformed - mean
        r_center = self.mx_r_inv @ center
        center_w = center * self.mx_w_inv
        return np.einsum('ji,ji->', center_w, r_center)

    def grad_reml2(self, x: np.ndarray):
        """
        Gradient of loglikelihood of matrix-variate normal distribution.

        For a second stage of REML estimation.
        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of MatNorm objective.

        """
        self.update_matrices(x)
        grad = np.zeros_like(x)
        center = self.mx_data_transformed - self.calc_mean(self.mx_m)
        t =  (center * self.mx_w_inv).T @ self.mx_r_inv
        mean_grad = self.calc_mean_grad_reml()
        for i, g in enumerate(mean_grad):
            if len(g.shape):
                grad[i] = -2 * np.einsum('ij,ji->', g, t)
        return grad

    '''
    ------------------Matrix Variate Normal Maximum Likelihood-----------------
    '''

    def obj_matnorm(self, x: np.ndarray):
        """
        Loglikelihood of matrix-variate normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood.

        """
        self.update_matrices(x)
        sigma, (m, _) = self.calc_sigma()
        try:
            r = self.calc_l(sigma)
            r_inv, logdet_r = chol_inv2(r)
            w_inv, logdet_w = self.calc_t_inv(sigma)
        except np.linalg.LinAlgError:
            return np.inf
        mean = self.calc_mean(m)
        center = self.mx_data_transformed - mean
        tr_r = np.trace(r)
        m, n = self.num_m, self.num_n
        r_center = r_inv @ center
        center_w = center * w_inv
        tr = tr_r * np.einsum('ij,ij->', center_w, r_center)
        return tr + m * logdet_w + n * logdet_r - n * m * np.log(tr_r)

    def grad_matnorm(self, x: np.ndarray):
        """
        Gradient of loglikelihood of matrix-variate normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of MatNorm objective.

        """
        self.update_matrices(x)
        grad = np.zeros_like(x)
        sigma, (m, c) = self.calc_sigma()
        try:
            r = self.calc_l(sigma)
            r_inv = chol_inv(r)
            w_inv, _ = self.calc_t_inv(sigma)
        except np.linalg.LinAlgError:
            grad[:] = np.inf
            return grad
        mean = self.calc_mean(m)
        center = self.mx_data_transformed - mean
        A = r_inv @ center
        B = (center * w_inv).T
        tr_ab = np.einsum('ij,ji->', A, B)
        tr_r = np.trace(r)
        V1 = center.T @ A
        V3 = B @ r_inv
        V2 = self.num_n * r_inv / tr_r - A @ V3

        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad(m, c)
        r_grad = self.calc_l_grad(sigma_grad)
        w_grad, w_grad_logdet = self.calc_t_inv_grad(w_inv, sigma_grad)
        n, m = self.num_n, self.num_m
        for i, (d_m, d_r, d_w, d_l) in enumerate(zip(mean_grad, r_grad,
                                                     w_grad, w_grad_logdet)):
            g = 0.0
            tr_long = 0.0
            if len(d_m.shape):
                tr_long -= 2 * np.einsum('ij,ji->', V3, d_m)
            if len(d_r.shape):
                tr_long += np.einsum('ij,ji->', V2, d_r)
                tr_dr = np.trace(d_r)
                g += tr_dr * tr_ab
                g -= m * n * tr_dr / tr_r
            if len(d_w.shape):
                tr_long += np.einsum('ii,i->', V1, d_w)
                g += m * d_l
            g += tr_r * tr_long
            grad[i] = g
        return grad

    '''
    -------------------------Best Linear Unbiased Predictor--------------------
    '''
    
    def calc_blup(self):
        """
        Estimate random effects values (BLUP).

        Returns
        -------
        np.ndarray
        Estimates of random effects.

        """
        sigma, (m, _) = self.calc_sigma()
        z = self.mx_data_transformed - self.calc_mean(m)
        d = self.mx_d
        inds = d.diagonal() < 1e-3
        inds_i = self.mx_s < 1e-4
        d = delete_mx(d, inds)
        sigma = delete_mx(sigma, inds) 
        z = np.delete(np.delete(z, inds, axis=0), inds_i, axis=1)
        sigma = np.linalg.inv(sigma)
        a = d @ sigma
        b = np.diag(self.mx_s[~inds_i] ** (-1))
        q = a @ z
        s = solve_sylvester(a, b, q)
        z = np.zeros((len(self.mx_s), len(d)))
        z[~inds_i, :] = s.T
        res = self.mx_q @ z
        cols = np.array(self.vars['observed'])[~inds]
        return pd.DataFrame(res, columns=cols)

    '''
    -----------------------Fisher Information Matrix---------------------------
    '''

    def calc_fim_ml(self, inverse=False):
        """
        Calculate Fisher Information Matrix.

        Exponential-family distributions are assumed.
        Parameters
        ----------
        inverse : bool, optional
            If True, function also returns inverse of FIM. The default is
            False.
        reml : bool, optional
            If True, then adjustment for REML is made. The default is False.

        Returns
        -------
        np.ndarray
            FIM.
        np.ndarray, optional
            FIM^{-1}.

        """
        sigma, (m, c) = self.calc_sigma()
        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad(m, c)
        t_inv = self.calc_t_inv(sigma)[0]
        l = self.calc_l(sigma)
        try:
            l_inv = chol_inv(l)
        except np.linalg.LinAlgError:
            l_inv = np.linalg.pinv(l)
        l_grad = self.calc_l_grad(sigma_grad)
        t_grad = self.calc_t_grad(sigma_grad)
        tr_l = np.trace(l)
        a = [t_inv * g if not np.isscalar(g) else None for g in t_grad]
        b = [l_inv @ g if not np.isscalar(g) else None for g in l_grad]
        tr_a = [np.sum(g) if g is not None else g for g in a]
        tr_b = [np.trace(g) if g is not None else g for g in b]
        m_t = [g * t_inv if not np.isscalar(g) else None for g in mean_grad]
        m_l = [g.T @ l_inv if not np.isscalar(g) else None for g in mean_grad]
        al = [np.trace(g) / tr_l if not np.isscalar(g) else None
              for g in l_grad]
        param_len = len(self.param_vals)
        fim = np.zeros((param_len, param_len))
        n = self.num_n
        m = self.num_m
        n, m = m, n
        for i in range(param_len):
            for j in range(i, param_len):
                mean = 0
                cov = 0
                mtj = m_t[j]
                mli = m_l[i]
                ai = a[i]
                aj = a[j]
                bi = b[i]
                bj = b[j]
                trai = tr_a[i]
                traj = tr_a[j]
                trbi = tr_b[i]
                trbj = tr_b[j]
                alphai = al[i]
                alphaj = al[j]
                if mli is not None and mtj is not None:
                    mean += tr_l * np.einsum('ij,ji', mli, mtj)
                if ai is not None and aj is not None:
                    cov += n * (ai * aj).sum()
                    cov += m * np.einsum('ij,ji', bi, bj)
                    cov += trai * trbj + trbi * traj
                    cov += n * m * alphai * alphaj
                    cov -= n * alphaj * trai + m * alphaj * trbi
                    cov -= n * alphai * traj + m * alphai * trbj
                fim[i, j] = mean + cov / 2
                fim[j, i] = fim[i, j]
        if inverse:
            fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim

    def calc_fim_means(self, inverse=False):
        """
        Calculate Fisher Information Matrix for mean components only.

        Exponential-family distributions are assumed. Useful to fascilate GWAS
        as we usually don't care about other parameters.
        Parameters
        ----------
        inverse : bool, optional
            If True, function also returns inverse of FIM. The default is
            False.

        Returns
        -------
        np.ndarray
            FIM.
        np.ndarray, optional
            FIM^{-1}.

        """
        sigma, (m, c) = self.calc_sigma()
        mean_grad = self.calc_mean_grad(m, c)
        t_inv = self.calc_t_inv(sigma)[0]
        l = self.calc_l(sigma)
        try:
            l_inv = chol_inv(l)
        except np.linalg.LinAlgError:
            l_inv = np.linalg.pinv(l)
        tr_l = np.trace(l)
        m_t = [g * t_inv if not np.isscalar(g) else None for g in mean_grad]
        m_l = [g.T @ l_inv if not np.isscalar(g) else None for g in mean_grad]
        inds = list()
        c = 0
        for g in m_t:
            if g is not None:
                inds.append(c)
                c += 1
            else:
                inds.append(None)
        fim_means = np.zeros((len(inds), len(inds)))
        param_len = len(self.param_vals)
        fim = np.zeros((param_len, param_len))
        for i in range(param_len):
            for j in range(i, param_len):
                mtj = m_t[j]
                mli = m_l[i]
                if mli is not None and mtj is not None:
                    fim[i, j] = tr_l * np.einsum('ij,ji', mli, mtj)
                    fim[j, i] = fim[i, j]
                    it, jt = inds[i], inds[j]
                    fim_means[it, jt] = fim[i, j]
                    fim_means[jt, it] = fim[i, j]
        if inverse:
            fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim

    '''
    -------------------------Prediction method--------------------------------
    '''
    
    def predict(self, x: pd.DataFrame, k=None, group=None):
        """
        Predict data given certain observations.
        
        Uses conditional expectation of the normal distribution method.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame with missing variables either not present at all, or
            with missing entries set to NaN.
        k : pd.DataFrame, optional
            K relatedness matrix for elements in x. If x is None, then
            observations are assumed to be independent. The default is None.
        group : str, optional
            Columns name where group ID is stored. The default is None.

        Returns
        -------
        None.

        """
        sigma, (m, _) = self.calc_sigma()
        obs = self.vars['observed']
        exo = self.vars['observed_exogenous']
        result = x.copy()
        for v in obs:
            if v not in result:
                result[v] = np.nan
        for v in exo:
            if v not in result:
                if v == '1':
                    result[v] = 1
                else:
                    result[v] = 0
        result = result[obs + exo]
        old_g = self.mx_g.copy()
        self.mx_g = result[exo].values.T
        mean = self.calc_mean(m).reshape((-1, 1), order='F')
        self.mx_g = old_g
        k = calc_zkz(x[group], k)
        l = sigma * len(x) + self.mx_d * np.trace(k)
        t = np.trace(sigma) * np.identity(len(x)) + k * np.trace(self.mx_d)
        l /= np.trace(l)
        cov = np.kron(t, l)
        data = result[obs].values.T
        data_shape = data.shape
        data = data.reshape((-1, 1), order='F')
        missing = np.isnan(data).flatten()
        present = ~missing
        cov12 = cov[missing][:, present]
        cov22 = np.linalg.pinv(cov[present][:, present])
        mean_m = mean[missing]
        mean_p = mean[present]
        preds = mean_m
        if len(present):
            preds += cov12 @ cov22 @ (data[present] - mean_p)
        data[missing] = preds
        data = data.reshape(data_shape, order='F').T
        result = pd.DataFrame(data, columns=obs, index=x.index)
        return result

    def predict_factors(self, x: pd.DataFrame):
        """
        Fast factor estimation method via MAP. Requires complete data.

        Parameters
        ----------
        x : pd.DataFrame
            Complete data of observed variables.

        Returns
        -------
        Factor scores.

        """
        lats = self.vars['latent']
        num_lat = len(lats)
        if num_lat == 0:
            return pd.DataFrame([])
        inners = self.vars['inner']
        obs = self.vars['observed']
        obs_exo = self.vars['observed_exogenous']
        g = []
        for v in obs_exo:
            if v == '1':
                g.append([1] * len(x))
            else:
                g.append(x[v])
        g = np.array(g)
        x = x[obs].values.T
        lambda_h = self.mx_lambda[:, :num_lat]
        lambda_x = self.mx_lambda[:, num_lat:]
        c = np.linalg.inv(np.identity(self.mx_beta.shape[0]) - self.mx_beta)
        c_1 = c[:num_lat, :]
        c_2 = c[num_lat:, :]
        g1 = self.mx_gamma1; g2 = self.mx_gamma2;
        M_h = x - (g2 + lambda_x @ c_2 @ g1) @ g
        t = lambda_x @ c_2
        L_zh = (t @ self.mx_psi @ t.T + self.mx_theta) * (x.shape[1])
        tr_sigma = np.trace(L_zh) / x.shape[1]
        L_zh += self.mx_d * np.trace(self.mx_zkz)
        tr_lzh = np.trace(L_zh)
        try:
            L_zh = chol_inv(L_zh)
        except np.linalg.LinAlgError:
            L_zh = np.linalg.pinv(L_zh)
        T_zh = np.identity(x.shape[1]) * tr_sigma
        T_zh += self.mx_zkz * np.trace(self.mx_d)
        try:
            T_zh = chol_inv(T_zh)
        except np.linalg.LinAlgError:
            T_zh = np.linalg.pinv(T_zh)
        t = lambda_h.T @ L_zh
        A = tr_lzh * t @ M_h @ T_zh
        A_0 = tr_lzh * t @ lambda_h
        try:
            L_h = chol_inv(c_1 @ self.mx_psi @ c_1.T)
        except np.linalg.LinAlgError:
            L_h = np.linalg.pinv(c_1 @ self.mx_psi @ c_1.T)
        M = c_1 @ g1 @ g
        A_1 = L_h @ M
        try:
            inv_A0 = chol_inv(A_0)
        except np.linalg.LinAlgError:
            inv_A0 = np.linalg.pinv(A_0)
        A_2 = inv_A0 @ L_h
        A_hat = inv_A0 @ (A + A_1)
        H = solve_sylvester(A_2, T_zh, A_hat)
        return pd.DataFrame(H.T, columns=filter(lambda v: v in lats, inners))