# -*- coding: utf-8 -*-
"""Generalized Random Effects SEM."""
from .utils import chol_inv, chol_inv2, cov
from scipy.linalg import solve_sylvester
from .model_means import ModelMeans
from itertools import combinations
from functools import partial
from . import startingvalues
from copy import deepcopy
import pandas as pd
import numpy as np


class ModelGeneralizedEffects(ModelMeans):
    """
    Generalized Random Effects model.
    
    Generalized Random Effects SEM is a generalization of ModelEffects in a
    sense, that it allows for an arbitrary number of random effects, and also
    it allows to introduce parametic covariance-between-observations marices.
    The latter can be thought of as in context of time-series or spatial data.
    """

    def __init__(self, description: str, effects: tuple, mimic_lavaan=False,
                 baseline=False, cov_diag=False, intercepts=True):
        """
        Instantiate Generalized Random Effects SEM model.

        Parameters
        ----------
        description : str
            Model description in semopy syntax.
        effects : tuple, EffectBase
            A list of Effects or a single effect.
        mimic_lavaan: bool
            If True, output variables are correlated and not conceptually
            identical to indicators. lavaan treats them that way, but it's
            less computationally effective. The default is False.
        baseline : bool
            If True, the model will be set to baseline model.
            Baseline model here is an independence model where all variables
            are considered to be independent with zero covariance. Only
            variances are estimated. The default is False.
        cov_diag : bool
            If cov_diag is True, then there are no covariances parametrised
            unless explicitly specified. The default is False.
        intercepts: bool
            If True, intercepts are also modeled. Intercept terms can be
            accessed via "1" symbol in a regression equation, i.e. x1 ~ 1. The
            default is False.

        Returns
        -------
        None.

        """
        if type(effects) not in (list, tuple):
            effects = (effects, )
        self.effects = effects
        self.symbs_rf = [f'~{i+1}~' for i in range(len(effects))]
        matrices = list(self.matrices_names)
        for i, symb in enumerate(self.symbs_rf):
            name = f'd{i+1}'
            setattr(self, f'build_{name}', self.build_d)
            setattr(self, f'start_{name}', startingvalues.start_d)
            matrices.append(name)
            f = partial(self.effect_rf_covariance, mx=name)
            self.dict_effects[symb] = f
        self.matrices_names = tuple(matrices)
        super().__init__(description, mimic_lavaan=False, baseline=baseline,
                         cov_diag=cov_diag, intercepts=intercepts)
        self.objectives = {'FIML': (self.obj_matnorm, self.grad_matnorm)}
    
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
        obs = self.vars['observed']
        for i, effect in enumerate(self.effects):
            symb = self.symbs_rf[i]
            mode = effect.d_mode
            if mode in ('diag', 'full'):
                for v in obs:
                    t = effects[symb][v]
                    if v not in t:
                        t[v] = None
                if mode == 'full':
                    for a, b in combinations(obs, 2):
                        t = effects[symb][a]
                        tt = effects[symb][b]
                        if (v not in t) and (v not in tt):
                            t[b] = None
            else:
                if mode != 'scale':
                    raise Exception(f'Unknown mode "{mode}".')
                param = f'paramD{i + 1}'
                for v in obs:
                    t = effects[symb][v][v] = param

    def load(self, data, cov=None, clean_slate=False, n_samples=None,
             **kwargs):
        """
        Load dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Data with columns as variables.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. Used only for variance
            starting values. The default is None.
        clean_slate : bool, optional
            If True, resets parameters vector. The default is False.
        n_samples : int, optional
            Redunant for ModelEffects. The default is None.
        **kwargs : dict
            Extra arguments are sent to Effects.

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
            raise KeyError(f'Variables {t} are missing from data.')
        self.load_data(data, covariance=cov, **kwargs)
        self.load_starting_values()
        if clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()


    def prepare_params(self):
        """
        Prepare structures for effective optimization routines.

        Returns
        -------
        None.

        """
        super().prepare_params()
        extra = np.array([])
        ranges = list()
        a = len(self.param_vals)
        for effect in self.effects:
            extra = np.append(extra, effect.parameters)
            b = a + len(effect.parameters)
            ranges.append((a, b))
            a = b
        self.param_vals = np.append(self.param_vals, extra)
        self.effects_param_ranges = ranges

    def update_matrices(self, params: np.ndarray):
        """
        Update all matrices from a parameter vector.

        Parameters
        ----------
        params : np.ndarray
            Vector of parameters.

        Returns
        -------
        None.

        """
        super().update_matrices(params)
        for effect, (a, b) in zip(self.effects, self.effects_param_ranges):
            effect.parameters[:] = params[a:b]

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

    def effect_rf_covariance(self, items: dict, mx: str):
        """
        Work through random effects covariance operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.
        mx : str
            Name of the D matrix.

        Returns
        -------
        None.

        """
        rows, cols = getattr(self, f'names_{mx}')
        mx = getattr(self, f'mx_{mx}')
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

    def get_bounds(self):
        """
        Get bound constraints on parameters.

        Returns
        -------
        list
            List of tuples specifying bounds.

        """
        b = super().get_bounds()
        for effect in self.effects:
            b.extend(effect.get_bounds())
        return b


    def fit(self, data=None, cov=None, obj='ML', solver='SLSQP', groups=None,
            clean_slate=False, **kwargs):
        """
        Fit model to data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data with columns as variables. The default is None.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. The default is None.
        obj : str, optional
            Objective function to minimize. Possible values are 'REML', "ML".
            The default is 'ML'.
        solver : TYPE, optional
            Optimizaiton method. Currently scipy-only methods are available.
            The default is 'SLSQP'.
        groups : list, optional
            Groups of size > 1 to center across. The default is None.
        clean_slate : bool, optional
            If False, successive fits will be performed with previous results
            as starting values. If True, parameter vector is reset each time
            prior to optimization. The default is False.

        Raises
        ------
        NotImplementedError
            Rises when unknown objective name is passed.

        Returns
        -------
        SolverResult
            Information on optimization process.

        """
        if obj == 'ML':
            res = super().fit(data=data, cov=cov, obj='ML', solver=solver,
                              groups=groups, clean_slate=clean_slate, **kwargs)
            return res
        else:
            raise NotImplementedError(f"Unknown method {obj}.")

    '''
    ---------------------Preparing structures for a more-----------------------
    ------------------------efficient computations-----------------------------
    '''

    def load_data(self, data: pd.DataFrame, covariance=None, **kwargs):
        """
        Load dataset from data matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with columns as variables and rows as observations.
        covariance : pd.DataFrame, optional 
            Custom covariance matrix. The default is None.
         **kwargs : dict
            Extra arguments are sent to Effects.

        Returns
        -------
        None.

        """
        obs = self.vars['observed']
        self.mx_g = data[self.vars['observed_exogenous']].values.T
        if len(self.mx_g.shape) != 2:
            self.mx_g = self.mx_g[np.newaxis, :]
        self.mx_data = data[obs].values.T
        self.n_obs, self.n_samples = self.mx_data.shape
        self.num_m = len(set(self.vars['observed']) - self.vars['latent'])
        self.load_cov(covariance[obs].loc[obs]
                      if covariance is not None else cov(self.mx_data.T))
        d_matrices = list()
        for i, effect in enumerate(self.effects):
            effect.load(i, self, data, **kwargs)
            d = getattr(self, f'mx_d{i + 1}')
            d_matrices.append(d)
        self.mxs_d = d_matrices
        self.mx_identity = np.identity(self.n_samples)

    '''
    ----------------------------LINEAR ALGEBRA PART---------------------------
    ----------------------The code below is responsible-----------------------
    ------------------for covariance structure computations-------------------
    '''

    def calc_l(self, sigma=None, k=None):
        """
        Calculate covariance across columns matrix T.
        
        Parameters
        ----------
        sigma: np.ndarray, optional
            Sigma covariance matrix as returned by calc_sigma. Although there
            is no meaningful concept of Sigma matrix in ModelEffects, it is
            still computationally convenient to separate it into an extra
            element. If None, then it will computed automatically. The default
            is None.
        k: tuple, optional
            List of K matrices as returned by calc_k by Effects. If None, then
            calculated in place. The default is None.
        Returns
        -------
        np.ndarray
            Covariance across columns (variables) matrix T.

        """
        if sigma is None:
            sigma, _ = self.calc_sigma()
        if k is None:
            k = self.calc_ks()
        n = self.n_samples
        return sum(np.trace(k) * d for d, k in zip(self.mxs_d, k)) + n * sigma

    def calc_l_grad(self, sigma=None, sigma_grad=None, k=None, k_grad=None):
        """
        Calculate gradient of covariance across columns matrix T.
        
        Parameters
        ----------
        sigma: np.ndarray, optional
            Sigma covariance matrix as returned by calc_sigma. Although there
            is no meaningful concept of Sigma matrix in ModelEffects, it is
            still computationally convenient to separate it into an extra
            element. If None, then it will computed automatically. The default
            is None.
        sigma_grad: List[np.ndarray], optional
            List of Sigma derivatives as returned by calc_sigma_grad. If None,
            then will be computed in place. The default is None.
        k: tuple, optional
            List of K matrices as returned by calc_k by Effects. If None, then
            calculated in place. The default is None.
        sigma_grad: List[List[np.ndarray]], optional
            List of K gradients as returned by calc_k_grad of Effect. If None,
            then will be computed in place. The default is None.
        Returns
        -------
        List[np.ndarray]
            List of derivatives ofcovariance across columns (variables) matrix
            wrt to model parameters.

        """
        if sigma is None:
            sigma, (m, c) = self.calc_sigma()
            if sigma_grad is None:
                sigma_grad = self.calc_sigma_grad(m, c)
        if k is None:
            k = self.calc_ks()
            if k_grad is None:
                k_grad= self.calc_ks_grad()
        k = list(map(np.trace, k))
        k_grad = list(map(np.trace, k_grad))
        grad = list()
        n = self.n_samples
        for g, df in zip(sigma_grad, self.mx_diffs):
            g = g * n
            for i in range(6, len(df)):
                if df[i] is not None:
                    g += df[i] * k[i - 6]
            grad.append(g)
        c = 0
        for i, (a, b) in enumerate(self.effects_param_ranges):
            d = self.mxs_d[i]
            for _ in range(b - a):
                grad.append(d * k_grad[c])
                c += 1
        return grad

    def calc_t(self, sigma=None, k=None):
        """
        Calculate covariance across rows matrix L.
        
        Parameters
        ----------
        sigma: np.ndarray, optional
            Sigma covariance matrix as returned by calc_sigma. Although there
            is no meaningful concept of Sigma matrix in ModelEffects, it is
            still computationally convenient to separate it into an extra
            element. If None, then it will computed automatically. The default
            is None.
        k: tuple, optional
            List of K matrices as returned by calc_k by Effects. If None, then
            calculated in place. The default is None.
        Returns
        -------
        np.ndarray
            Covariance across rows (observations) matrix L.

        """
        if sigma is None:
            sigma, _ = self.calc_sigma()
        if k is None:
            k = self.calc_ks()
        s = self.mx_identity * np.trace(sigma)
        return sum(np.trace(d) * k for d, k in zip(self.mxs_d, k)) + s

    def calc_t_grad(self, sigma=None, sigma_grad=None, k=None, k_grad=None):
        """
        Calculate gradient of covariance across rows matrix L.
        
        Parameters
        ----------
        sigma: np.ndarray, optional
            Sigma covariance matrix as returned by calc_sigma. Although there
            is no meaningful concept of Sigma matrix in ModelEffects, it is
            still computationally convenient to separate it into an extra
            element. If None, then it will computed automatically. The default
            is None.
        sigma_grad: List[np.ndarray], optional
            List of Sigma derivatives as returned by calc_sigma_grad. If None,
            then will be computed in place. The default is None.
        k: tuple, optional
            List of K matrices as returned by calc_k by Effects. If None, then
            calculated in place. The default is None.
        sigma_grad: List[List[np.ndarray]], optional
            List of K gradients as returned by calc_k_grad of Effect. If None,
            then will be computed in place. The default is None.
        Returns
        -------
        List[np.ndarray]
            List of derivatives ofcovariance across rows (observations) matrix
            wrt to model parameters.

        """
        if sigma is None:
            sigma, (m, c) = self.calc_sigma()
            if sigma_grad is None:
                sigma_grad = self.calc_sigma_grad(m, c)
        if k is None:
            k = self.calc_ks()
            if k_grad is None:
                k_grad= self.calc_ks_grad()
        grad = list()
        for g, df in zip(sigma_grad, self.mx_diffs):
            try:
                g = np.trace(g) * self.mx_identity
            except ValueError:
                g = 0.0
            for i in range(6, len(df)):
                if df[i] is not None:
                    g += np.trace(df[i]) * k[i - 6]
            grad.append(g)
        c = 0
        for i, (a, b) in enumerate(self.effects_param_ranges):
            d = np.trace(self.mxs_d[i])
            for _ in range(b - a):
                grad.append(d * k_grad[c])
                c += 1
        return grad

    def calc_ks(self):
        return [effect.calc_k(self) for effect in self.effects]

    def calc_ks_grad(self):
        grad = list()
        for effect in self.effects:
            grad.extend(effect.calc_k_grad(self))
        return grad
    
    def calc_mean_grad(self, m=None, c=None):
        if m is None:
            m, c = self.calc_sigma()[1]
        grad = super().calc_mean_grad(m, c)
        n = len(grad)
        p = len(self.param_vals)
        grad.extend([np.float(0.0)] * (p - n))
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
            Loglikelihood (constants omitted).

        """
        self.update_matrices(x)
        sigma, (m, _) = self.calc_sigma()
        k = self.calc_ks()
        l = self.calc_l(sigma, k)
        t = self.calc_t(sigma, k)
        try:
            l_inv, l_logdet = chol_inv2(l)
            t_inv, t_logdet = chol_inv2(t)
        except np.linalg.LinAlgError:
            return np.nan
        center = self.mx_data - self.calc_mean(m)
        tr_l = np.trace(l)
        a = tr_l * np.einsum('ij,ji', l_inv @ center, t_inv @ center.T)
        m = self.num_m
        n = self.n_samples
        return a + n * l_logdet + m * t_logdet - n * m * np.log(tr_l)
        
    def grad_matnorm(self, x: np.ndarray):
        grad = np.zeros_like(x)
        self.update_matrices(x)
        sigma, (m, c) = self.calc_sigma()
        k = self.calc_ks()
        try:
            l = self.calc_l(sigma, k)
            l_inv = chol_inv(l)
            t = self.calc_t(sigma, k)
            t_inv = chol_inv(t)
        except np.linalg.LinAlgError:
            grad[:] = np.inf
            return grad
        mean_grad = self.calc_mean_grad(m, c)
        sigma_grad = self.calc_sigma_grad(m, c)
        k_grad = self.calc_ks_grad()
        l_grad = self.calc_l_grad(sigma, sigma_grad, k, k_grad)
        t_grad = self.calc_t_grad(sigma, sigma_grad, k, k_grad)
        center = self.mx_data - self.calc_mean(m)
        m = self.num_m
        n = self.n_samples
        c0 = t_inv @ center.T @ l_inv
        # a1 = l_inv @ center @ t_inv
        c1 = c0 @ center
        c2 = center @ c0
        big_tr = np.trace(c1)
        tr_l = np.trace(l)
        for i, (m_g, l_g, t_g) in enumerate(zip(mean_grad, l_grad, t_grad)):
            g = 0.0
            if not np.isscalar(m_g):
                g -= 2 * tr_l * np.einsum('ij,ji', c0, m_g)
            if not np.isscalar(l_g):
                ai = t_inv @ t_g
                bi = l_g @ l_inv
                tr_lg = np.trace(l_g)
                g += tr_lg * big_tr + m * np.trace(ai) + n * np.trace(bi)
                g -= tr_l * (np.einsum('ij,ji', ai, c1) + \
                             np.einsum('ij,ji', bi, c2))
                g -= n * m * tr_lg / tr_l 
            grad[i] = g
        return grad

    '''
    -------------------------Prediction method--------------------------------
    '''
    
    def predict(self, x: pd.DataFrame, effects=None):
        """
        Predict data given certain observations.
        
        Uses conditional expectation of the normal distribution method.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame with missing variables either not present at all, or
            with missing entries set to NaN.
        k : List[EfffectBase], optional
            List of effects to be used for prediction. If None, effects that
            were used during optimization. The default is None.

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
        if effects is None:
            effects = deepcopy(self.effects)
        for i, effect in enumerate(effects):
            effect.load(i, self, x, clean_start=False)
        k = [effect.calc_k(self) for effect in effects]
        old_n = self.n_samples
        old_i = self.mx_identity
        self.n_samples = len(x)
        self.mx_identity = np.identity(self.n_samples)
        l = self.calc_l(sigma, k)
        t = self.calc_t(sigma, k)
        l = l / np.trace(l)
        self.n_samples = old_n
        self.mx_identity = old_i
        cov = np.kron(t, l)
        data = result[obs].values.T
        data_shape = data.shape
        data = data.reshape((-1, 1), order='F')
        missing = np.isnan(data).flatten()
        present = ~missing
        cov12 = cov[missing][:, present]
        cov22 = np.linalg.inv(cov[present][:, present])
        mean_m = mean[missing]
        mean_p = mean[present]
        preds = mean_m
        if len(present):
            preds = mean_m + cov12 @ cov22 @ (data[present] - mean_p)
        data[missing] = preds
        result = data.reshape(data_shape, order='F').T
        result = pd.DataFrame(result, columns=obs)
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
        ks = [effect.calc_k(self) for effect in self.effects]
        ds = self.mxs_d
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
        L_zh += sum(d * np.trace(k) for d, k in zip(ds, ks))
        tr_lzh = np.trace(L_zh)
        try:
            L_zh = chol_inv(L_zh)
        except np.linalg.LinAlgError:
            L_zh = np.linalg.pinv(L_zh)
        T_zh = np.identity(x.shape[1]) * tr_sigma
        T_zh += sum(k * np.trace(d) for d, k in zip(ds, ks))
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

    '''
    -------------------------Best Linear Unbiased Predictor--------------------
    '''
    
    def calc_blup(self, ind_effects=None):
        """
        Estimate random effects values (BLUP).

        Parameters
        ----------
        ind_effects : List[int], optional
            Indices of random effects that will be included to the estimate.
            If None, then all the effects will be estimated and summed up.
            The default is None.

        Returns
        -------
        np.ndarray
        Estimates of random effects.

        """
        if ind_effects is None:
            ind_effects = list(range(len(self.effects)))
        left_effects = list()
        l_i = 0
        t_i = 0
        for i, effect in enumerate(self.effects):
            k = effect.calc_k(self)
            d = self.mxs_d[i]
            if i in ind_effects:
                left_effects.append(np.zeros_like(k))
                t_i += k * np.trace(d)
                l_i += d * np.trace(k)
            else:
                left_effects.append(k)
        sigma, (m, _) = self.calc_sigma()
        z = self.mx_data - self.calc_mean(m)
        t = self.calc_t(sigma, left_effects)
        l = self.calc_l(sigma, left_effects)
        l = l / np.trace(l)
        l_i = l_i / np.trace(l_i)
        t_inv = np.linalg.inv(t)
        t_i_inv = np.linalg.inv(t_i)
        l_i_inv = np.linalg.inv(l_i)
        a = l @ l_i_inv
        b = t_inv @ t_i_inv
        q = z @ t_inv @ t_i
        return solve_sylvester(a, b, q)

    '''
    -------------------------Fisher Information Matrix------------------------
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
        ks = [effect.calc_k(self) for effect in self.effects]
        k_grad = self.calc_ks_grad()
        t = self.calc_t(sigma, ks)
        l = self.calc_l(sigma, ks)
        try:
            t_inv = chol_inv(t)
        except np.linalg.LinAlgError:
            t_inv= np.linalg.pinv(t)
        try:
            l_inv = chol_inv(l)
        except np.linalg.LinAlgError:
            l_inv = np.linalg.pinv(l)
        l_grad = self.calc_l_grad(sigma, sigma_grad, ks, k_grad)
        t_grad = self.calc_t_grad(sigma, sigma_grad, ks, k_grad)
        tr_l = np.trace(l)
        a = [t_inv @ g if not np.isscalar(g) else None for g in t_grad]
        b = [l_inv @ g if not np.isscalar(g) else None for g in l_grad]
        tr_a = [np.trace(g) if g is not None else g for g in a]
        tr_b = [np.trace(g) if g is not None else g for g in b]
        m_t = [g @ t_inv if not np.isscalar(g) else None for g in mean_grad]
        m_l = [g.T @ l_inv if not np.isscalar(g) else None for g in mean_grad]
        al = [np.trace(g) / tr_l if not np.isscalar(g) else None
              for g in l_grad]
        param_len = len(self.param_vals)
        fim = np.zeros((param_len, param_len))
        n = self.n_samples
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
                    cov += n * np.einsum('ij,ji', ai, aj)
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
