#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mean estimator for Model without meanstructure. FOR INTERNAL USAGE ONLY."""

import pandas as pd
import numpy as np
from .model import Model
from .solver import Solver
from .utils import chol_inv
from .stats import calc_se, calc_zvals, calc_pvals


class MeanEstimator():
    """Simple mean estimator for the Model without meanstructure."""

    __slots__ = ['sigma', 'm', 'mx_data', 'mx_data_t', 'diffs', 'range_mu',
                 'range_vu', 'mu', 'vu', 'param_vals', 'last_result',
                 'sigma_inv', 'vars']

    def __init__(self, model: Model):
        """
        Instantiate mean estimator.

        Parameters
        ----------
        model : Model
            Model without meanstructure.

        Raises
        ------
        Exception
            Raises when no data is available.

        Returns
        -------
        None.

        """
        if not hasattr(model, 'mx_data'):
            raise Exception('Can''t estimate means if no data is available.')
        self.sigma, (self.m, _) = model.calc_sigma()
        self.sigma_inv = chol_inv(self.sigma)
        self.mx_data = model.mx_data
        self.mx_data_t = model.mx_data.T
        self.vars = dict()
        obs = model.vars['observed']
        self.vars['observed'] = obs
        set_obs = set(obs)
        inner = model.vars['inner']
        outer = model.vars['_output']
        self.vars['inner'] = inner
        self.vars['_output'] = outer
        self.mu = np.zeros((len(inner), 1))
        self.vu = np.zeros((len(obs), 1))
        obs_inner = set_obs & set(inner)
        inner = [i for i, v in enumerate(inner) if v in set_obs]
        innert = [obs.index(v) for v in obs if v in obs_inner]
        outer = [i for i, v in enumerate(obs) if v in outer]
        self.mu[inner, 0] = np.nanmean(self.mx_data_t[:, innert], axis=0)
        self.vu[outer, 0] = np.nanmean(self.mx_data_t[:, outer], axis=0)
        self.range_mu = (inner, (0, len(inner)))
        self.range_vu = (outer, (len(inner), len(inner) + len(outer)))
        self.build_diff_vectors()
        self.param_vals = list(self.mu[inner].flatten())
        self.param_vals += list(self.vu[outer].flatten())
        self.param_vals = np.array(self.param_vals)

    def build_diff_vectors(self):
        """
        Calculate derivatives of mean vectors.

        Returns
        -------
        None.

        """
        diffs = list()
        mt = -self.m
        for i in self.range_mu[0]:
            diffs.append(mt[:, i][:, np.newaxis])
        for i in self.range_vu[0]:
            t = np.zeros_like(self.vu)
            t[i, 0] = -1
            diffs.append(t)
        self.diffs = diffs

    def fit(self, obj='ML', solver='SLSQP'):
        """
        Estimate means.

        Parameters
        ----------
        obj : str, optional
            Objective function to minimize. Possible values are 'ML', 'ULS'.
            'GLS' is same as 'ML'. The default is 'MLW'.
        solver : TYPE, optional
            Optimizaiton method. Currently scipy-only methods are available.
            The default is 'SLSQP'.

        Raises
        ------
        Exception
            Rises when attempting to use FIML in absence of full data.

        Returns
        -------
        Optimization result.

        """
        fun, grad = self.get_objective(obj)
        solver = Solver(solver, fun, grad, self.param_vals)
        res = solver.solve()
        res.name_obj = obj
        self.param_vals = res.x
        self.update_vectors(res.x)
        self.last_result = res
        return res

    def get_objective(self, name: str):
        """
        Retrieve objective function and its gradient by name.

        Parameters
        ----------
        name : str
            Name of objective function.

        Raises
        ------
        KeyError
            Rises if incorrect name is provided.

        Returns
        -------
        tuple
            Objective function and gradient function.

        """
        d = {'ML': (self.obj_ml, self.grad_ml),
             'GLS': (self.obj_ml, self.grad_ml),
             'ULS': (self.obj_uls, self.grad_uls)}
        try:
            return d[name]
        except KeyError:
            raise KeyError(f'{name} is unknown objective function.')

    def update_vectors(self, x: np.ndarray):
        """
        Update mean vectors from vector of parameters.

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.range_mu[0]:
            self.mu[self.range_mu[0], 0] = x[:self.range_mu[1][1]]
        if self.range_vu[0]:
            self.vu[self.range_vu[0], 0] = x[self.range_vu[1][0]:]

    def calc_center(self):
        """
        Calculate centered data.

        Returns
        -------
        np.ndarray
            Data with means subtracted.

        """
        return self.mx_data_t - self.m @ self.mu - self.vu

    def calc_center_grad(self):
        """
        Calculate center gradient.

        Returns
        -------
        np.ndarray
            Centered gradient.

        """
        return self.diffs

    def calc_fim(self, inverse=False):
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
        center_grad = self.calc_center_grad()
        sz = len(center_grad)
        info = np.zeros((sz, sz))
        sgs = [sg.T @ self.sigma_inv for sg in center_grad]
        for i in range(sz):
            for k in range(i, sz):
                info[i, k] = np.einsum('ij,ji->', sgs[i], center_grad[k])
        fim = info + np.triu(info, 1).T
        if inverse:
            fim_inv = chol_inv(fim)
            return (fim, fim_inv)
        return fim

    def obj_ml(self, x: np.ndarray):
        """
        Calculate gaussian maximum likelihood objective function.

        Here it is the same as GLS.
        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood value.

        """
        self.update_vectors(x)
        center = self.calc_center()
        return np.einsum('ij,ji->', center.T, self.sigma_inv @ center)

    def grad_ml(self, x: np.ndarray):
        """
        Calculate maximum likelihood objective function gradient.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Array of derivatives at point x .

        """
        self.update_vectors(x)
        t = self.calc_center().T @ self.sigma_inv
        res = np.zeros(x.shape)
        df = self.calc_center_grad()
        for i in range(len(res)):
            res[i] = 2 * np.einsum('ij,ji->', t, df[i])
        return res

    def obj_uls(self, x: np.ndarray):
        """
        Calculate gaussian maximum likelihood objective function.

        Here it is the same as GLS.
        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood value.

        """
        self.update_vectors(x)
        center = self.calc_center()
        return np.einsum('ij,ji->', center.T, center)

    def grad_uls(self, x: np.ndarray):
        """
        Calculate ULS objective function gradient.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Array of derivatives at point x .

        """
        self.update_vectors(x)
        t = self.calc_center().T
        res = np.zeros(x.shape)
        df = self.calc_center_grad()
        for i in range(len(res)):
            res[i] = 2 * np.einsum('ij,ji->', t, df[i])
        return res


def estimate_means(mod: Model, method='ML', solver='SLSQP', 
                   pvals=False, ret_opt=False):
    """
    Estimate means for meanstructure-free model.

    Mean estimation is performed via fitting mean components to data given
    known covariance matrix Sigma. In case of ML/GLS, it's equivalent to
    minimizing Mahalanobis distance with respect to mean components. Usually,
    output variables estimates are really close to just data mean estimates.
    Parameters
    ----------
    mod : Model
        Model.
    method : str, optional
        Can 'ML'/'GLS', 'ULS'. The default is 'ML'.
    solver : str, optional
        Solver to use. The default is 'SLSQP'.
    pvals : bool, optional
        If True, then p-values are returned. They posses no information for
        most purposes here and usually are close to 1.0. The default is False.
    ret_opt : bool, optional
        If True, otpimizer info is also returned. The default is False.

    Returns
    -------
    pd.DataFrame
        Information on mean components.
    SolverResult, optional
        Information on optimization result.
    """
    me = MeanEstimator(mod)
    res = me.fit(method, solver)
    df = list()
    i = 0
    if pvals:
        se = calc_se(me)
        zv = calc_zvals(me, std_errors=se)
        pv = calc_pvals(me, z_scores=zv)
    else:
        se = np.zeros_like(res.x)
        zv = se
        pv = zv
    for v in mod.vars['inner']:
        if v in mod.vars['observed']:
            df.append((v, '~', '1', res.x[i], se[i], zv[i], pv[i]))
            i += 1
    for v in mod.vars['observed']:
        if v in mod.vars['_output']:
            df.append((v, '~', '1', res.x[i], se[i], zv[i], pv[i]))
            i += 1
    df = pd.DataFrame(df, columns=['lval', 'op', 'rval', 'Estimate',
                                   'Std. Err', 'z-value', 'p-value'])
    if not pvals:
        df = df.drop(['Std. Err', 'z-value', 'p-value'], axis=1)
    return df if not ret_opt else (df, res)
