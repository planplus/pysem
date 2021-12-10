#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistic and fit indices."""
from .utils import duplication_matrix
from scipy.stats import norm, chi2
from collections import namedtuple
from .optimizer import Optimizer
from .model_generalized_effects import ModelGeneralizedEffects
from .model_effects import ModelEffects
from .model_means import ModelMeans
from .model import Model
import pandas as pd
import numpy as np

ParameterStatistics = namedtuple('ParametersStatistics',
                                 ['value', 'se', 'zscore', 'pvalue'])
SEMStatistics = namedtuple('SEMStatistics', ['dof', 'ml', 'fun',
                                             'chi2', 'dof_baseline',
                                             'chi2_baseline', 'rmsea', 'cfi',
                                             'gfi', 'agfi', 'nfi', 'tli',
                                             'aic', 'bic', 'params'])


def get_baseline_model(model, data=None):
    """
    Retrieve a the baseline model from given model.

    Baseline model here is an independence model where all variables are
    considered to be independent with zero covariance. Only variances are
    estimated.
    Parameters
    ----------
    model : Model
        Model.
    data : pd.DataFrame, optional
        Data, extracting from model will be attempted if no data provided.
        (It's assumed that model.load took place). The default is None.
    cov : pd.DataFrame, optional
        Covariance matrix can be provided in the lack of presense of data.
        The default is None.

    Returns
    -------
    mod : Model
        Baseline model.

    """
    if type(model) is str:
        mod = Model(model, baseline=True)
        if data:
            mod.load(data)
        return mod
    desc = model.description
    tp = type(model)
    tp_name = tp.__name__
    if tp_name not in ('Model', ):
        mod = tp(desc, baseline=True, intercepts=model.intercepts)
    else:
        mod = tp(desc, baseline=True)
    try:
        if not data:
            if hasattr(model, 'mx_data'):
                mx = model.mx_data
                vs = model.vars['observed']
                if hasattr(model, 'mx_g'):
                    mx = np.append(mx, model.mx_g.T, axis=1)
                    vs = vs + model.vars['observed_exogenous']
                data = pd.DataFrame(data=mx, columns=vs)
                if hasattr(model, 'group'):
                    data[model.group] = model.group_data[0, :]
                if tp_name in ('ModelEffects',):
                    mod.load(data, group=model.group, k=model.passed_k)
                else:
                    mod.load(data)
            else:
                data = pd.DataFrame(data=model.mx_cov,
                                    index=model.vars['observed'],
                                    columns=model.vars['observed']) 
                mod.load(cov=data, n_samples=model.n_samples)
    except AttributeError:
        pass
    return mod


def __get_chi2_base(model):
    """
    Calculate chi2 of baseline model.

    Parameters
    ----------
    model : Model
        Model.

    Returns
    -------
    tuple
        chi2, dof of baseline model.

    """
    mod_base = get_baseline_model(model)
    obj = model.last_result.name_obj
    if type(mod_base).__name__ in ('ModelEffects', 'ModelGeneralizedEffects'):
        if obj == 'REML2':
            obj = 'REML'
        elif obj == 'FIML':
            obj = 'ML'
    elif type(mod_base).__name__ in ('ModelMeans', ):
        if obj == 'FIML':
            obj = 'ML'
    mod_base.fit(obj=obj)
    chi2_base = calc_chi2(mod_base)[0]
    return chi2_base, calc_dof(mod_base)


def calc_gfi(model, chi2=None, chi2_base=None):
    """
    Calculate GFI (goodness-of-fit index).

    Parameters
    ----------
    model : model
        Model.
    chi2 : TYPE, optional
        chi2 statistic. The default is None.
    chi2_base : TYPE, optional
        chi2 statistic for baseline model. The default is None.

    Returns
    -------
    float
        GFI.

    """
    if chi2 is None:
        chi2 = calc_chi2(model)[0]
    if chi2_base is None:
        chi2_base = __get_chi2_base(model)[0]
    return 1 - chi2 / chi2_base


def calc_agfi(model, dof=None, dof_bs=None, gfi=None):
    """
    Calculate AGFI (adjusted goodness-of-fit index).

    Parameters
    ----------
    model : Model
        Model.
    dof : int, optional
        Degrees of freedom. The default is None.
    dof_bs : float, optional
        Degrees of freedom of baseline model. The default is None.
    gfi : float, optional
        GFI statistic. The default is None.

    Returns
    -------
    float
        AGFI.

    """
    if dof is None:
        dof = calc_dof(model)
    if dof == 0:
        return np.nan
    if dof_bs is None:
        base = get_baseline_model(model)
        dof_bs = calc_dof(base)
    if gfi is None:
        gfi = calc_gfi(model)
    return 1 - dof_bs / dof * (1 - gfi)  # k * (k + 1) / (2 * dof) * (1 - gfi)


def calc_dof(model):
    """
    Calculate degrees of freedom.

    Parameters
    ----------
    model : Model
        Model.

    Returns
    -------
    int
        DoF.

    """
    p = len(model.vars['observed'])
    return p * (p + 1) // 2 - len(model.param_vals)


def calc_nfi(model, chi2=None, chi2_base=None):
    """
    Calculate NFI (Normed Fit Index).

    Parameters
    ----------
    model : Model
        Model.
    chi2 : float, optional
        chi2 statistic. The default is None.
    chi2_base : TYPE, optional
        chi2 statistic of baseline model. The default is None.

    Returns
    -------
    float
        NFI.

    """
    if chi2 is None:
        chi2 = calc_chi2(model)[0]
    if chi2_base is None:
        chi2_base = __get_chi2_base(model)[0]
    if chi2_base == 0:
        return np.nan
    return (chi2_base - chi2) / chi2_base


def calc_tli(model, dof=None, chi2=None, dof_base=None, chi2_base=None):
    """
    Calculate TLI (Tucker and Lewis Index).

    Parameters
    ----------
    model : Model
        Model.
    dof : int, optional
        Degrees of freedom statistic. The default is None.
    chi2 : float, optional
        chi2 statistic. The default is None.
    dof_base : int, optional
        Degrees of freedom of baseline model. The default is None.
    chi2_base : float, optional
        chi2 statistic of baseline model. The default is None.

    Returns
    -------
    float
        TLI.

    """
    if chi2 is None:
        chi2 = calc_chi2(model)[0]
    if chi2_base is None or dof_base is None:
        chi2_base, dof_base = __get_chi2_base(model)
    if dof is None:
        dof = calc_dof(model)
    if dof == 0 or dof_base == 0:
        return np.nan
    a, b = chi2 / dof, chi2_base / dof_base
    return (b - a) / (b - 1)


def calc_cfi(model, dof=None, chi2=None, dof_base=None, chi2_base=None):
    """
    Calculate CFI (Comparative Fit Index).

    Parameters
    ----------
    model : Model
        Model.
    dof : int, optional
        Degrees of freedom statistic. The default is None.
    chi2 : float, optional
        chi2 statistic. The default is None.
    dof_base : int, optional
        Degrees of freedom statistic of baseline model. The default is None.
    chi2_base : float, optional
        chi2 statistic of baseline model. The default is None.

    Returns
    -------
    float
        CFI.

    """
    if chi2 is None:
        chi2 = calc_chi2(model)[0]
    if dof is None:
        dof = calc_dof(model)
    if chi2_base is None or dof_base is None:
        chi2_base, dof_base = __get_chi2_base(model)
    a = chi2 - dof
    b = chi2_base - dof_base
    return 1 - a / b


def calc_chi2(model, dof=None):
    """
    Calculate chi-square statistic.

    Parameters
    ----------
    model : Model
        Model.
    dof : int, optional
        Degrees of freedom statistic. The default is None.

    Returns
    -------
    tuple
        chi2 statistic and p-value.

    """
    if dof is None:
        dof = calc_dof(model)
    if model.last_result.name_obj == 'FIML':
        stat = model.last_result.fun / model.n_samples
    else:
        stat = model.n_samples * model.last_result.fun
    return stat, 1 - chi2.cdf(stat, dof)


def calc_chi2_sb(model, stat=None, dof=None):
    """
    Calculate scaled (Satorra-Bentler) chi2.

    Parameters
    ----------
    model : Model
        Model.
    stat : float, optional
        chi2 statistic. The default is None.
    dof : int, optional
        Degrees of freedom. The default is None.

    Returns
    -------
    float
        SB chi2.
    float
        p-value.

    """
    if stat is None:
        stat = calc_chi2(model)[0]
    if dof is None:
        dof = calc_dof(model)
    sigma, (m, c) = model.calc_sigma()
    inds = np.triu_indices_from(sigma)
    grad = np.array([g[inds].flatten(order='F')
                    for g in model.calc_sigma_grad(m, c)]).T
    sigma_inv = np.linalg.inv(sigma)
    k = np.kron(sigma_inv, sigma_inv)
    dup = duplication_matrix(sigma.shape[0])
    w = dup.T @ k @ dup / 2
    w = np.linalg.inv(w)
    t = w @ grad
    u = w - t @ np.linalg.inv(grad.T @ t) @ t.T
    if hasattr(model, 'mx_w'):
        c = np.einsum('ij,ji->', u, model.mx_w) / dof
    else:
        c = np.trace(u @ np.linalg.inv(w)) / dof
        return t, grad, w, u
    stat /= c
    return stat, 1 - chi2.cdf(stat, dof)


def calc_rmsea(model, chi2=None, dof=None):
    """
    Calculate RMSEA statistic.

    Parameters
    ----------
    model : Model
        Model.
    chi2 : float, optional
        chi2 statistic. The default is None.
    dof : int, optional
        Degrees of freedom statistic. The default is None.

    Returns
    -------
    float
        RMSEA.

    """
    if chi2 is None:
        chi2 = calc_chi2(model)[0]
    if dof is None:
        dof = calc_dof(model)
    if chi2 < dof:
        return 0
    return np.sqrt((chi2 / dof - 1) / (model.n_samples - 1))


def calc_likelihood(model):
    """
    Calculate likelihood.

    Parameters
    ----------
    model : Model
        Model.

    Returns
    -------
    float
        Loglikelihood.

    """
    # TODO
    return model.obj_mlw(model.param_vals)


def calc_aic(model, lh=None):
    """
    Calculate Akaike's Information Criteria (AIC) statistic.

    Parameters
    ----------
    model : Model
        Model.
    lh : float, optional
        Loglikelihood. The default is None.

    Returns
    -------
    float
        AIC.

    """
    if lh is None:
        lh = calc_likelihood(model)
    return 2 * (len(model.param_vals) - lh)


def calc_bic(model, lh=None):
    """
    Calculate Bayesian Information Criteria (BIC) statistic.

    Parameters
    ----------
    model : Model
        Model.
    lh : float, optional
        Loglikelihood. The default is None.

    Returns
    -------
    float
        BIC.

    """
    if lh is None:
        lh = calc_likelihood(model)
    k, n = len(model.param_vals), model.n_samples
    return np.log(n) * k - 2 * lh


def calc_se(model, information='expected', robust=False):
    """
    Calculate standard errors.

    Parameters
    ----------
    model : Model
        Model.
    information : str
        If 'expected', expected Fisher information is used. Otherwise,
        observed information is employed. The default is 'expected'.
    robust : bool, optional
        If True, then robust SE are computed instead. Robustness here means
        that MLR-esque sandwich correction is applied. The default is False.

    Returns
    -------
    list
        list of standard errors of model's active parameters.

    """
    if robust or information != 'expected':
        # An explanation of "mult" for a reader:
        # When I was writing code and deriving some formulas for different
        # objective functions, I was finding myself quiet often dropping
        # constants when they appeared to be not necessary. Sometimes I didn't.
        # It is a mess. Here I am taking that in account. Some may say that
        # that I'd better just get those goddamn constants back in place
        # and get rid of this clusterfuck, and they'd be right, but I am
        # currently too hesitant to do it myself. 
        # My sincere apologies.
        # Georgy
        mult = model.n_samples / 2
        if hasattr(model, 'last_result'):
            fun, grad = model.get_objective(model.last_result.name_obj)
            if model.last_result.name_obj in ('FIML'):
                mult = 0.5
        else:
            try:
                fun, grad = model.get_objective('MatNorm')
            except KeyError:
                try:
                    fun, grad = model.get_objective('FIML')
                except KeyError:
                    fun, grad = model.get_objective('MLW')
                    mult = 1.0
        
    if information == 'expected':
        mx_inf = model.calc_fim(inverse=True)[1]
    else:
        from numdifftools import Gradient, Hessian
        if grad is None:
            hess = Hessian(fun)(model.param_vals)
        else:
            hess = Gradient(grad)(model.param_vals)
        mx_inf = np.linalg.pinv(hess) / mult
    if robust:
        g = sum(np.outer(g,g) for g in model.grad_se_g(model.param_vals))
        mx_inf = mx_inf @ g @ mx_inf
    variances = mx_inf.diagonal().copy()
    inds = (variances < 0) & (variances > -1e-1)
    variances[inds] = 1e-12
    variances[variances < 0] = np.nan  # So numpy won't throw a warning.
    return np.sqrt(variances) 


def calc_zvals(model, std_errors=None, information='expected'):
    """
    Calculate Z-score.

    Parameters
    ----------
    model : Model
        Model.
    std_errors : list, optional
        Standard errors. The default is None.
    information : str
        If 'expected', expected Fisher information is used. Otherwise,
        observed information is employed. No use if std_errors are provided.
        The default is 'expected'.

    Returns
    -------
    list
        Z-scores.

    """
    if std_errors is None:
        std_errors = calc_se(model, information=information)
    return [val / (std + 1e-12) for val, std in zip(list(model.param_vals),
                                                    std_errors)]


def calc_pvals(model, z_scores=None, information='expected'):
    """
    Calculate p-values.

    Parameters
    ----------
    model : Model
        Model.
    z_scores : list, optional
        Z-sores. The default is None.
    information : str
        If 'expected', expected Fisher information is used. Otherwise,
        observed information is employed. No use if std_errors are provided.
        The default is 'expected'.

    Returns
    -------
    list
        P-values.

    """
    if z_scores is None:
        z_scores = calc_zvals(model, information=information)
    return [2 * (1 - norm.cdf(abs(z))) for z in z_scores]


def calc_stats(model):
    """
    Calculate fit indices and other auxiliary statistics.

    Parameters
    ----------
    model : Model
        Fit model instance.

    Returns
    -------
    pd.DataFrame
        Fit indices and statistics.

    """
    if not hasattr(model, 'last_result'):
        raise Exception('Can''t gather statitics from model until it is fit.')
    try:
        lh = calc_likelihood(model)
    except np.linalg.LinAlgError:
        lh = np.nan
    aic = calc_aic(model, lh)
    bic = calc_bic(model, lh)
    dof = calc_dof(model)
    chi2_base, dof_base = __get_chi2_base(model)
    chi2 = calc_chi2(model, dof)
    rmsea = calc_rmsea(model, chi2[0], dof)
    cfi = calc_cfi(model, dof, chi2[0], dof_base, chi2_base)
    gfi = calc_gfi(model, chi2[0], chi2_base)
    agfi = calc_agfi(model, dof, dof_base, gfi)
    nfi = calc_nfi(model, chi2[0], chi2_base)
    tli = calc_tli(model, dof, chi2[0], dof_base, chi2_base)
    d = {'DoF': dof, 'DoF Baseline': dof_base,
         'chi2': chi2[0], 'chi2 p-value': chi2[1], 'chi2 Baseline': chi2_base,
         'CFI': cfi, 'GFI': gfi, 'AGFI': agfi, 'NFI': nfi, 'TLI': tli,
         'RMSEA': rmsea, 'AIC': aic, 'BIC': bic, 'LogLik': lh}
    return pd.DataFrame(d, index=['Value'])


def gather_statistics(model, information='expected', robust=False):
    """
    Retrieve all statistics as specified in SEMStatistics structure.

    Needed for backwards-compaitability with semopy 1.+.
    Parameters
    ----------
    model : Model
        Model.
    information : str
        If 'expected', expected Fisher information is used. Otherwise,
        observed information is employed. The default is 'expected'.
    robust : bool, optional
        If True, then robust SE are computed instead. Robustness here means
        that MLR-esque sandwich correction is applied. The default is False.

    Raises
    ------
    Exception
        Rises when no data is loaded into the model.

    Returns
    -------
    SEMStatistics
        Namedtuple containing all statistics available to semopy.

    """
    if isinstance(model, Optimizer):
        model = model.model
    if not hasattr(model, 'last_result'):
        raise Exception('Can''t gather statitics from model until it is fit.')
    values = model.param_vals.copy()
    try:
        std_errors = calc_se(model, information=information, robust=robust)
    except np.linalg.LinAlgError:
        std_errors = np.array([np.nan] * len(values))
    z_scores = calc_zvals(model, std_errors)
    pvalues = calc_pvals(model, z_scores)
    try:
        lh = calc_likelihood(model)
    except np.linalg.LinAlgError:
        lh = np.nan
    aic = calc_aic(model, lh)
    bic = calc_bic(model, lh)
    paramStats = [ParameterStatistics(val, std, ztest, pvalue)
                  for val, std, ztest, pvalue
                  in zip(values, std_errors, z_scores, pvalues)]
    dof = calc_dof(model)
    chi2_base, dof_base = __get_chi2_base(model)
    chi2 = calc_chi2(model, dof)
    rmsea = calc_rmsea(model, chi2[0], dof)
    cfi = calc_cfi(model, dof, chi2[0], dof_base, chi2_base)
    gfi = calc_gfi(model, chi2[0], chi2_base)
    agfi = calc_agfi(model, dof, dof_base, gfi)
    nfi = calc_nfi(model, chi2[0], chi2_base)
    tli = calc_tli(model, dof, chi2[0], dof_base, chi2_base)
    return SEMStatistics(dof, model.last_result.fun, lh, chi2, dof_base,
                         chi2_base, rmsea, cfi, gfi, agfi, nfi, tli, aic, bic,
                         paramStats)
