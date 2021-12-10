#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Augmented semopy model with mean structure."""
import logging
import numpy as np
import pandas as pd
from .model import Model
from collections import defaultdict
from scipy.linalg import block_diag
from .utils import chol, chol_inv, chol_inv2, delete_mx, cov


class ModelMeans(Model):
    """
    Model with a mean stucture.

    Augmented model with exogenous variables ruled out into a separate
    structure. This, in turn, results in all exogenous variables being de-facto
    absent from the model, which may lead to less computations in certain
    cases (i.e. when a number of observed exogenous variables is huge, for
    example in GWAS).
    """

    matrices_names = tuple(list(Model.matrices_names) + ['gamma1', 'gamma2'])

    def __init__(self, description: str, mimic_lavaan=False, baseline=False,
                 cov_diag=False, intercepts=True):
        """
        Instantiate Model with mean-structure.

        Parameters
        ----------
        description : str
            Model description in semopy syntax.
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
            default is True.

        Returns
        -------
        None.

        """
        self.intercepts = intercepts
        self.calc_fim = self.calc_fim_ml
        super().__init__(description, mimic_lavaan=mimic_lavaan,
                         cov_diag=cov_diag, baseline=baseline)
        self.objectives = {'FIML': (self.obj_fiml, self.grad_fiml),
                           'REML': (self.obj_reml, self.grad_reml),
                           'GLS': (self.obj_gls, self.grad_gls)}


    def preprocess_effects(self, effects: dict):
        """
        Run a routine just before effects are applied.

        Used to apply covariances to model.
        Parameters
        -------
        effects : dict
            Mapping opcode->lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        super().preprocess_effects(effects)
        if self.intercepts:
            for v in self.vars['observed']:
                if v not in self.vars['latent']:  # Workaround for Imputer
                    t = effects[self.symb_regression][v]
                    if '1' not in t:
                        t['1'] = None
        gamma1 = set()  # Those that load onto inner variables
        gamma2 = set()  # Those that load onto output variables
        for lval, rvs in effects[self.symb_regression].items():
            rvals = filter(lambda v: v in self.vars['observed_exogenous'], rvs)
            if lval in self.vars['_output']:
                gamma2.update(rvals)
            else:
                gamma1.update(rvals)
        self.vars['observed_exogenous'] = sorted(gamma1 | gamma2)
        self.vars['observed_exogenous_1'] = gamma1
        self.vars['observed_exogenous_2'] = gamma2

    def finalize_variable_classification(self, effects: dict):
        """
        Finalize variable classification.

        Reorders variables for better visual fancyness and does extra
        model-specific variable respecification.
        
        Parameters
        -------
        effects : dict
            Maping opcode->values->rvalues->mutiplicator.

        Returns
        -------
        None.

        """
        obs = self.vars['observed']
        exo = self.vars['exogenous']
        if self.intercepts:
            exo.add('1')
            obs.add('1')
        obs_exo = obs & exo
        to_rem = set()
        covs = effects[self.symb_covariance]
        for lv, rvs in covs.items():
            to_rem.add(lv)
            to_rem.update(rvs)
        obs_exo -= to_rem
        obs -= obs_exo
        exo -= obs_exo
        self.vars['observed_exogenous'] = obs_exo
        super().finalize_variable_classification(effects)

    def build_gamma1(self):
        """
        Gamma1 matrix contains relationships with exogenous variables.

        This Gamma1 matrix loads onto INNER variables, i.e. non-output
        variables.
        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        rows, cols = self.vars['inner'], self.vars['observed_exogenous']
        n, m = len(rows), len(cols)
        mx = np.zeros((n, m))
        return mx, (rows, cols)

    def build_gamma2(self):
        """
        Gamma2 matrix contains relationships with exogenous variables.

        This Gamma2 matrix loads onto OUTPUT variables.
        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        rows, cols = self.vars['observed'], self.vars['observed_exogenous']
        n, m = len(rows), len(cols)
        mx = np.zeros((n, m))
        return mx, (rows, cols)

    def prepare_fiml(self):
        """
        Prepare data structure for efficient FIML calculation.

        Returns
        -------
        None.

        """
        d = defaultdict(list)
        data = self.mx_data
        for i in range(data.shape[0]):
            t = tuple(list(np.where(np.isfinite(data[i]))[0]))
            d[t].append(i)
        for cols, rows in d.items():
            inds = tuple(i for i in range(data.shape[1])
                         if i not in cols)
            t = data[rows, :][:, cols]
            d[cols] = (t.T, list(inds), rows)
        self.fiml_data = d

    def effect_regression(self, items: dict):
        """
        Work through regression operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        items_super = defaultdict(dict)
        exo_obs = self.vars['observed_exogenous']
        inner = self.vars['inner']
        baseline = self.baseline
        for lv, rvs in items.items():
            for rv, mult in rvs.items():
                if baseline and rv != '1':
                    continue
                if rv not in exo_obs:
                    items_super[lv][rv] = mult
                    continue
                if lv in inner:
                    rows, cols = self.names_gamma1
                    i = rows.index(lv)
                    j = cols.index(rv)
                    mx = self.mx_gamma1
                else:
                    rows, cols = self.names_gamma2
                    i = rows.index(lv)
                    j = cols.index(rv)
                    mx = self.mx_gamma2
                ind = (i, j)
                name = None
                active = True
                try:
                    val = float(mult)
                    active = False
                except (TypeError, ValueError):
                    if mult is not None:
                        if mult == self.symb_starting_values:
                            active = False
                        else:
                            name = mult
                    val = None
                if name is None:
                    self.n_param_reg += 1
                    name = '_b%s' % self.n_param_reg
                self.add_param(name=name, matrix=mx, indices=ind, start=val,
                               active=active, symmetric=False,
                               bound=(None, None))
        super().effect_regression(items_super)

    def operation_define(self, operation):
        """
        Works through DEFINE command.

        Here, used to prevent user from attempting to use ordinal variables.
        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Returns
        -------
        None.

        """
        if operation.params and operation.params[0] == 'ordinal':
            raise SyntaxWarning("Models with mean component do not support \
                                ordinal variables.")
        super().operation_define(operation)

    def load_data(self, data: pd.DataFrame, covariance=None, groups=None):
        """
        Load dataset from data matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with columns as variables and rows as observations.
        covariance : pd.DataFrame, optional
            Custom covariance matrix. The default is None.
        groups : list, optional
            List of group names to center across. The default is None.

        Returns
        -------
        None.

        """
        if groups is None:
            groups = list()
        obs = self.vars['observed']
        for group in groups:
            for g in data[group].unique():
                inds = data[group] == g
                if sum(inds) == 1:
                    continue
                data.loc[inds, obs] -= data.loc[inds, obs].mean()
                data.loc[inds, group] = g
        self.mx_data = data[obs].values
        if len(self.mx_data.shape) != 2:
            self.mx_data = self.mx_data[:, np.newaxis]
        self.n_samples, self.n_obs = self.mx_data.shape
        self.mx_g = data[self.vars['observed_exogenous']].values.T
        if len(self.mx_g.shape) != 2:
            self.mx_g = self.mx_g[np.newaxis, :]
        g = self.mx_g
        if self.calc_fim == self.calc_fim_reml:
            try:
                s = np.identity(g.shape[1]) - g.T @ chol_inv(g @ g.T) @ g
            except ValueError:
                s = np.identity(g.shape[1]) - g.T @ g
            d, q = np.linalg.eigh(s)
            rank_dec = 0
            for i in d:
                if abs(i) < 1e-8:
                    rank_dec += 1
                else:
                    break
            d = np.diag(d)[rank_dec:, :]
            self.mx_s = d @ q.T
            self.mx_data_transformed = self.mx_s @ self.mx_data
            self.mx_data_square = self.mx_data_transformed.T @\
                                  self.mx_data_transformed
        self.load_cov(covariance.loc[obs, obs]
                      if covariance is not None else cov(self.mx_data))

    def load(self, data, cov=None, groups=None, clean_slate=False,
             n_samples=None):
        """
        Load dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Data with columns as variables.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. Used only for variance
            starting values. The default is None.
        groups : list, optional
            Groups of size > 1 to center across. The default is None.
        clean_slate : bool, optional
            If True, resets parameters vector. The default is False.
        n_samples : int, optional
            Redunant for ModelMeans. The default is None.

        KeyError
            Rises when there are missing variables from the data.

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
        missing = (set(obs) | set(exo)) - set(cols)
        if missing:
            t = ', '.join(missing)
            raise KeyError('Variables {} are missing from data.'.format(t))
        self.load_data(data, covariance=cov, groups=groups)
        self.load_starting_values()
        if clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()
        # Happens only if we ran fit with ML
        if self.calc_fim is self.calc_fim_ml:
            self.prepare_fiml()

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
        Exception
            Rises when attempting to use FIML in absence of full data.
        NotImplementedError
            Rises when unknown objective name is passed.

        Returns
        -------
        SolverResult
            Information on optimization process.

        """
        if obj == 'REML':
            self.calc_fim = self.calc_fim_reml
            res = super().fit(data=data, cov=cov, obj='REML', solver=solver,
                              groups=groups, clean_slate=clean_slate, **kwargs)
            sigma, (self.mx_m, self.mx_c) = self.calc_sigma()
            self.mx_sigma_inv = chol_inv(sigma)
            res_m = super().fit(obj='GLS', solver=solver,
                                groups=groups, clean_slate=False, **kwargs)
            return res, res_m
        elif obj == 'ML':
            self.calc_fim = self.calc_fim_ml
            res = super().fit(data=data, cov=cov, obj='FIML', solver=solver,
                              groups=groups, clean_slate=clean_slate, **kwargs)
            return res
        else:
            raise NotImplementedError(f"Unknown method {obj}.")

    '''
    ----------------------------LINEAR ALGEBRA PART---------------------------
    ----------------------The code below is responsible-----------------------
    ---------------------for mean structure computations------===-------------
    '''

    def calc_mean(self, m: np.ndarray):
        """
        Calculate mean component.

        Parameters
        ----------
        m : np.ndarray
            Lambda @ C.

        Returns
        -------
        np.ndarray
            Model-implied mean component.

        """
        return (m @ self.mx_gamma1 + self.mx_gamma2) @ self.mx_g

    def calc_mean_grad(self, m: np.ndarray, c: np.ndarray):
        """
        Calculate mean component gradient.

        Parameters
        ----------
        m : np.ndarray
            Lambda @ C.
        c : np.ndarray
            (I-B)^{-1}.

        Returns
        -------
        grad : list
            Gradient values of model-implied mean component.

        """
        grad = list()
        gm1_g1 = self.mx_gamma1 @ self.mx_g
        c_gm1_g1 = c @ gm1_g1
        for dmx in self.mx_diffs:
            g = np.float32(0.0)
            if dmx[0] is not None:  # Beta
                g += m @ dmx[0] @ c_gm1_g1
            if dmx[1] is not None:  # Lambda
                g += dmx[1] @ c_gm1_g1
            if dmx[4] is not None:  # Gamma1
                g += m @ dmx[4] @ self.mx_g
            if dmx[5] is not None:  # Gamma2
                g += dmx[5] @ self.mx_g
            grad.append(g)
        return grad

    def calc_mean_grad_reml(self):
        """
        Calculate mean component gradient given Sigma.

        Returns
        -------
        grad : list
            Gradient values of model-implied mean component.

        """
        grad = list()
        m = self.mx_m
        for dmx in self.mx_diffs:
            g = np.float32(0.0)
            if dmx[4] is not None:  # Gamma1
                g += m @ dmx[4] @ self.mx_g
            if dmx[5] is not None:  # Gamma2
                g += dmx[5] @ self.mx_g
            grad.append(g)
        return grad

    '''
    -----------------------Restricted Maximum Likelihood-----------------------
    '''

    def obj_gls(self, x: np.ndarray):
        """
        GLS objective for fitting mean component given known Sigma.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            GLS obective value.

        """
        self.update_matrices(x)
        sigma_inv = self.mx_sigma_inv
        m = self.mx_m
        mean = self.calc_mean(m)
        mx = self.mx_data
        center = mx - mean.T
        return np.einsum('ij,ji->', center.T @ center, sigma_inv)

    def grad_gls(self, x: np.ndarray):
        """
        Calculate GLS gradient.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of GLS.

        """
        self.update_matrices(x)
        sigma_inv = self.mx_sigma_inv
        m = self.mx_m
        mean = self.calc_mean(m)
        mx = self.mx_data
        center = mx - mean.T
        t = center @ sigma_inv
        grad = list()
        for g in self.calc_mean_grad_reml():
            if len(g.shape):
                grad.append(-2 * np.einsum('ij,ji', g, t))
            else:
                grad.append(0.0)
        return np.array(grad)

    def obj_reml(self, x: np.ndarray):
        """
        Calculate REML objective function.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            REML.

        """
        self.update_matrices(x)
        sigma, _ = self.calc_sigma()
        tr = 0
        logdet = 0
        try:
            sigma_inv, logdet_sigma = chol_inv2(sigma)
        except np.linalg.LinAlgError:
            return np.inf
        tr += np.einsum('ij,ji->', self.mx_data_square, sigma_inv)
        logdet += self.mx_data_transformed.shape[0] * logdet_sigma
        loss = tr + logdet
        if loss < 0:
            return np.inf
        return loss

    def grad_reml(self, x: np.ndarray):
        """
        Calculate REML gradient.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of REML.

        """
        self.update_matrices(x)
        sigma, (m, c) = self.calc_sigma()
        sigma_grad = self.calc_sigma_grad(m, c)
        try:
            sigma_inv = chol_inv(sigma)
        except np.linalg.LinAlgError:
            return np.array([np.inf] * len(x))
        n = self.mx_data_transformed.shape[0]
        cs = n * sigma_inv - sigma_inv @ self.mx_data_square @ sigma_inv
        grad = list()
        for dx in sigma_grad:
            if len(dx.shape):
                grad.append(np.einsum('ij,ji->', cs, dx))
            else:
                grad.append(0.0)
        return np.array(grad)

    '''
    ---------------------Full Information Maximum Likelihood-------------------
    '''

    def obj_fiml(self, x: np.ndarray):
        """
        Calculate FIML objective function.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            FIML.

        """
        self.update_matrices(x)
        sigma_full, (m, _) = self.calc_sigma()
        mean_full = self.calc_mean(m)
        tr = 0
        logdet = 0
        for _, (mx, inds, rows) in self.fiml_data.items():
            center = mx - np.delete(mean_full, inds, axis=0)[:, rows]
            s = center @ center.T
            sigma = delete_mx(sigma_full, inds)
            try:
                sigma_inv, logdet_sigma = chol_inv2(sigma)
            except np.linalg.LinAlgError:
                return np.inf
            tr += np.einsum('ij,ji->', s, sigma_inv)
            logdet += len(rows) * logdet_sigma
        loss = tr + logdet
        if loss < 0:  # Realistically should never happen.
            return np.inf
        return loss

    def grad_fiml(self, x: np.ndarray):
        """
        Calculate FIML gradient.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of FIML.

        """
        self.update_matrices(x)
        sigma_full, (m, c) = self.calc_sigma()
        mean_full = self.calc_mean(m)
        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad(m, c)
        grad = [0.0] * len(sigma_grad)
        for _, (mx, inds, rows) in self.fiml_data.items():
            sigma = delete_mx(sigma_full, inds)
            try:
                sigma_inv = chol_inv(sigma)
            except np.linalg.LinAlgError:
                t = np.zeros(len(grad))
                t[:] = np.inf
                return t
            center = mx - np.delete(mean_full, inds, axis=0)[:, rows]
            s = center @ center.T
            a = center.T @ sigma_inv
            t = sigma_inv @ s @ sigma_inv
            n = len(rows)
            for i, (s_g, m_g) in enumerate(zip(sigma_grad, mean_grad)):
                if len(m_g.shape):
                    m_g = np.delete(m_g, inds, axis=0)[:, rows]
                    mean_tr = -2 * np.einsum('ij,ji->', m_g, a)
                else:
                    mean_tr = 0.0
                if len(s_g.shape):
                    s_g = delete_mx(s_g, inds)
                    tr = -np.einsum('ij,ji->', t, s_g)
                    logdet_tr = n * np.einsum('ij,ji->', sigma_inv, s_g)
                else:
                    logdet_tr = 0.0
                    tr = 0.0
                grad[i] += tr + logdet_tr + mean_tr
        return np.array(grad)

    '''
    -------------------------Prediction method--------------------------------
    '''

    def predict(self, x: pd.DataFrame):
        """
        Predict data given certain observations.
        
        Uses conditional expectation of the normal distribution method.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame with missing variables either not present at all, or
            with missing entries set to NaN.

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
        old_gamma = self.mx_g.copy()
        for i, (_, row) in enumerate(result.iterrows()):
            row_exo = row.loc[exo]
            row_endo = row.loc[obs]
            present = [True if np.isfinite(row[r]) else False
                       for r in obs]
            present = np.array(present)
            missing = ~present
            sigma12 = sigma[missing][:, present]
            sigma22 = np.linalg.pinv(sigma[present][:, present])
            self.mx_g = row_exo.values.reshape((-1, 1))
            mean = self.calc_mean(m)
            mean_m = mean[missing, :]
            mean_p = mean[present, :]
            p = mean_m
            if len(present):
                r = row_endo.iloc[present].values.reshape((-1, 1))
                p += sigma12 @ sigma22 @ (r - mean_p)
            row.iloc[np.where(missing)] = p.flatten()
            result.iloc[i] = row
        self.mx_g = old_gamma
        if '1' in result:
            result = result.drop('1', axis=1)
        return result

    def predict_exo(self, exogenous: pd.DataFrame):
        """
        Predict output variables given a set of exogenous variables.

        This method works much faster than "predict", however it can't be
        utilised to impute missing data or to estimate factors as of now. It is
        especially useful for phenotype prediction via a set of known SNPs.
        Parameters
        ----------
        exogenous : pd.DataFrame
            Observations of exogenous variables. Missing variables or mssing
            data are converted to zeros.

        Returns
        -------
        pd.DataFrame
            DataFrame containing predictions of endogenous observed variables.

        """
        exos = self.vars['observed_exogenous']
        exogenous = exogenous.copy()
        for v in exos:
            if v not in exogenous.columns:
                if v != '1':
                    exogenous[v] = 0
                else:
                    exogenous[v] = 1
        g = exogenous[exos].values.T 
        t = np.linalg.inv(np.identity(self.mx_beta.shape[0]) - self.mx_beta)
        t = (self.mx_lambda @ t @ self.mx_gamma1 + self.mx_gamma2) @ g
        return pd.DataFrame(t.T, columns=self.vars['observed'],
                            index=exogenous.index)

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
        x = x[obs].values.T
        lambda_h = self.mx_lambda[:, :num_lat]
        lambda_x = self.mx_lambda[:, num_lat:]
        c = np.linalg.inv(np.identity(self.mx_beta.shape[0]) - self.mx_beta)
        c_1 = c[:num_lat, :]
        c_2 = c[num_lat:, :]
        g1 = self.mx_gamma1; g2 = self.mx_gamma2; g = self.mx_g
        M_h = x - (g2 + lambda_x @ c_2 @ g1) @ g
        t = lambda_x @ c_2
        L_zh = (t @ self.mx_psi @ t.T + self.mx_theta) * (x.shape[1])
        tr_lzh = np.trace(L_zh)
        tr_sigma = tr_lzh / x.shape[1]
        L_zh = np.linalg.inv(L_zh)
        t = lambda_h.T @ L_zh
        A = (tr_lzh / tr_sigma) * t @ M_h
        A_0 = tr_lzh * t @ lambda_h
        L_h = np.linalg.inv(c_1 @ self.mx_psi @ c_1.T)
        M = c_1 @ g1 @ g
        A_1 = L_h @ M
        H = np.linalg.inv(A_0 / tr_sigma + L_h) @ (A_1 + A)
        return pd.DataFrame(H.T, columns=filter(lambda v: v in lats, inners))

    '''
    -------------------------Fisher Information Matrix------------------------
    '''

    def calc_fim_reml(self, inverse=False):
        """
        Calculate Fisher Information Matrix when estimation was performed via
        REML.

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
        if not hasattr('self', 'mx_sigma_inv'):
            sigma, (m, c) = self.calc_sigma()
            sigma_inv = chol_inv(sigma)
        else:
            sigma_inv, (m, c) = self.mx_sigma_inv, self.mx_m, self.mx_c
        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad_reml()
        n = self.mx_data.shape[0] / 2
        inds_mean = list()
        inds_sigma = list()
        sgs, mgs = list(), list()
        for i, g in enumerate(sigma_grad):
            if len(g.shape):
                sgs.append(g @ sigma_inv)
                inds_sigma.append(i)
        for i, g in enumerate(mean_grad):
            if len(g.shape):
                mgs.append(sigma_inv @ g)
                inds_mean.append(i)
        sz = len(sgs)
        mx_var = np.zeros((sz, sz))
        for i in range(sz):
            for j in range(i, sz):
                mx_var[i, j] = n * np.einsum('ij,ji->', sgs[i], sgs[j])
        mx_var = mx_var + np.triu(mx_var, 1).T
        sz = len(mgs)
        mx_fixed = np.zeros((sz, sz))
        for i in range(sz):
            for j in range(i, sz):
                mx_fixed[i, j] = np.einsum('ij,ij->', mean_grad[inds_mean[i]],
                                           mgs[j])
        mx_fixed = mx_fixed + np.triu(mx_fixed, 1).T
        inds_mean = np.array(inds_mean, dtype=int)
        inds_sigma = np.array(inds_sigma, dtype=int)
        inds = np.append(inds_mean, inds_sigma)
        fim = block_diag(mx_fixed, mx_var)
        fim = fim[:, inds][:, inds]
        if inverse:
            try:
                mx_var_inv = chol_inv(mx_var)
                mx_fixed_inv = chol_inv(mx_fixed)
                self._fim_warn = False
            except np.linalg.LinAlgError:
                logging.warn("Fisher Information Matrix is not PD."
                             "Moore-Penrose inverse will be used instead of "
                             "Cholesky decomposition. See "
                              "10.1109/TSP.2012.2208105.")
                self._fim_warn = True
                mx_var_inv = np.linalg.pinv(mx_var)
                mx_fixed_inv = np.linalg.pinv(mx_fixed)
            fim_inv = block_diag(mx_fixed_inv, mx_var_inv)
            fim_inv = fim_inv[inds, :][:, inds]
            return (fim, fim_inv)
        return fim

    def calc_fim_ml(self, inverse=False):
        """
        Calculate Fisher Information Matrix when estimation was performed via
        ML.

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
        inv_sigma = chol_inv(sigma)
        sz = len(sigma_grad)
        n = self.mx_data.shape[0] / 2
        info = np.zeros((sz, sz))
        sgs = [sg @ inv_sigma if len(sg.shape) else None for sg in sigma_grad]
        mgs = [inv_sigma @ g if len(g.shape) else None for g in mean_grad]
        for i in range(sz):
            for k in range(i, sz):
                if sgs[i] is not None and sgs[k] is not None:
                    info[i, k] = n * np.einsum('ij,ji->', sgs[i], sgs[k])
                if mgs[i] is not None and len(mean_grad[k].shape):
                    info[i, k] += np.einsum('ij,ij->', mean_grad[i], mgs[k])
        fim = info + np.triu(info, 1).T
        if inverse:
            try:
                fim_inv = chol_inv(fim)
                self.fim_warn = False
            except np.linalg.LinAlgError:
                logging.warn("Fisher Information Matrix is not PD."
                             "Moore-Penrose inverse will be used instead of "
                             "Cholesky decomposition. See "
                              "10.1109/TSP.2012.2208105.")
                self._fim_warn = True
                fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim


    def grad_se_g(self, x: np.ndarray):
        """
        Calculate a list of separate likelihoods for each observation.

        A helper function that might be used to estimate Huber-White sandwich
        corrections.
        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        list
            List of len n_samples.

        """
        self.update_matrices(x)
        try:
            sigma, (m, c) = self.calc_sigma()
            sigma_grad = self.calc_sigma_grad(m, c)
            mean_grad = self.calc_mean_grad(m, c)
            mean = self.calc_mean(m).T
            inv_sigma = np.linalg.pinv(sigma)
        except np.linalg.LinAlgError:
            t = np.zeros((len(x),))
            t[:] = np.inf
            return t
        res = list()
        mx_i = np.identity(sigma.shape[0])
        data = self.mx_data.copy()
        if not self.intercepts:
            data -= data.mean(axis=0)
        for i in range(self.mx_data.shape[0]):
            x = self.mx_data[i] - mean[i]
            x = x[:, np.newaxis]
            t = inv_sigma @ (mx_i -  x @ x.T @ inv_sigma)
            t2 = (inv_sigma @ x).flatten()
            g = np.zeros_like(self.param_vals)
            for i, (s_g, m_g) in enumerate(zip(sigma_grad, mean_grad)):
                if len(m_g.shape):
                    m = np.dot(t2, m_g[:, i])
                else:
                    m = 0.0
                if len(s_g.shape):
                    cov = np.einsum('ij,ji->', t, s_g) / 2
                else:
                    cov = 0
                g[i] += m + cov
            res.append(g)      
        return res