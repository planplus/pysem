#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements polychoric and polyserial correlations."""
from statsmodels.stats.correlation_tools import corr_nearest
from scipy.optimize import minimize, minimize_scalar
from itertools import chain, product, combinations
from scipy.stats import norm, mvn
from .utils import cor
import pandas as pd
import numpy as np


def bivariate_cdf(lower, upper, corr, means=[0, 0], var=[1, 1]):
    """
    Estimates an integral of bivariate pdf.

    Estimates an integral of bivariate pdf given integration lower and 
    upper limits. Consider using relatively big (i.e. 20 if using default mean
    and variance) lower and/or upper bounds when integrating to/from infinity.
    Parameters
    ----------
    lower : float
        Lower integration bounds.
    upper : float
        Upper integration bounds.
    corr : float
        Correlation coefficient between variables.
    means : list, optional
        Mean values of variables. The default is [0, 0].
    var : list, optional
        Variances of variables. The default is [1, 1].

    Returns
    -------
    float
        P(lower[0] < x < upper[0], lower[1] < y < upper[1]).

    """
    s = np.array([[var[0], corr], [corr, var[1]]])
    return mvn.mvnun(lower, upper, means, s)[0]

def univariate_cdf(lower, upper, mean=0, var=1):
    """
    Estimate an integral of univariate pdf.

    Estimate an integral of univariate pdf given integration lower and 
    upper limits. Consider using relatively big (i.e. 20 if using default mean
    and variance) lower and/or upper bounds when integrating to/from infinity.
    Parameters
    ----------
    lower : float
        Lower integration bound..
    upper : float
        Upper integration bound..
    mean : float, optional
        Mean value of the variable. The default is 0.
    var : float, optional
        Variance of the variable. The default is 1.

    Returns
    -------
    float
        P(lower < x < upper).

    """
    return mvn.mvnun([lower], [upper], [mean], [var])[0]

def estimate_intervals(x, inf=10):
    """
    Estimate intervals of the polytomized underlying latent variable.

    Parameters
    ----------
    x : np.ndarray
        An array of values the ordinal variable..
    inf : float, optional
        A numerical infinity substitute. The default is 10.

    Returns
    -------
    np.ndarray
        An array containing polytomy intervals.
    np.ndarray
        An array containing indices of intervals corresponding to each entry.

    """
    x_f = x[~np.isnan(x)]
    u, counts = np.unique(x_f, return_counts=True)
    sz = len(x_f)
    cumcounts = np.cumsum(counts[:-1])
    u = [np.where(u == sample)[0][0] + 1 for sample in x]
    return list(chain([-inf], (norm.ppf(n / sz) for n in cumcounts), [inf])), u

def polyserial_corr(x, y, x_mean=None, x_var=None, x_z=None, x_pdfs=None,
                    y_ints=None, scalar=True):
    """
    Estimate polyserial correlation.
    
    Estimate polyserial correlation between continious variable x and
    ordinal variable y.
    Parameters
    ----------
    x : np.ndarray
        Data sample corresponding to x.
    y : np.ndarray
        Data sample corresponding to ordinal variable y.
    x_mean : float, optional
        Mean of x (calculate if not provided). The default is None.
    x_var : float, optional
        Variance of x (calculate if not provided). The default is None.
    x_z : np.ndarray, optional
        Stabdardized x (calculated if not provided). The default is None.
    x_pdfs : np.ndarray, optional
        x's logpdf sampled at each point (calculated if not provided). The
        default is None.
    y_ints : list, optional
        Polytomic intervals of an underlying latent variable
        correspoding to y (calculated if not provided) as returned by
        estimate_intervals.. The default is None.
    scalar : bool, optional
        If true minimize_scalar is used instead of SLSQP.. The default is True.

    Returns
    -------
    float
        A polyserial correlation coefficient for x and y..

    """
    if x_mean is None:
        x_mean = np.nanmean(x)
    if x_var is None:
        x_var = np.nanvar(x)
    if y_ints is None:
        y_ints = estimate_intervals(y)
    if x_z is None:
        x_z = (x - x_mean) / x_var
    if x_pdfs is None:
        x_pdfs = norm.logpdf(x, x_mean, x_var)
    ints, inds = y_ints
    def transform_tau(tau, rho, z):
        return (tau - rho * z) / np.sqrt(1 - rho ** 2)
    def sub_pr(k, rho, z):
        i = transform_tau(ints[k], rho, z)
        j = transform_tau(ints[k - 1], rho, z)
        return univariate_cdf(j, i)
    def calc_likelihood(rho):
        return -sum(pdf + np.log(sub_pr(ind, rho, z))
                    for z, ind, pdf in zip(x_z, inds, x_pdfs))
    def calc_likelihood_derivative(rho):
        def sub(k, z):
            i = transform_tau(ints[k], rho, z)
            j = transform_tau(ints[k - 1], rho, z)
            a = norm.pdf(i) * (ints[k] * rho - z)
            b = norm.pdf(j) * (ints[k - 1] * rho - z)
            return a - b
        t = (1 - rho ** 2) ** 1.5
        return -sum(sub(ind, z) / sub_pr(ind, rho, z)
                   for x, z, ind in zip(x, x_z, inds) if not np.isnan(x)) / t
    if not scalar:
        res = minimize(calc_likelihood, [0.0], jac=calc_likelihood_derivative,
                       method='SLSQP', bounds=[(-1.0, 1.0)]).x[0]
    else:
        res = minimize_scalar(calc_likelihood, bounds=(-1, 1),
                              method='bounded').x
    return res

def polychoric_corr(x, y, x_ints=None, y_ints=None):
    """
    Estimate polyserial correlation between ordinal variables x and y.

    Parameters
    ----------
    x : np.ndarray
        Data sample corresponding to x.
    y : np.ndarray
        Data sample corresponding to y.
    x_ints : list, optional
        Polytomic intervals of an underlying latent variable correspoding to y
        (calculated if not provided) as returned by estimate_intervals. The
        default is None.
    y_ints : list, optional
        Polytomic intervals of an underlying latent variable correspoding to y
        (calculated if not provided) as returned by estimate_intervals. The
        default is None.

    Returns
    -------
    float
        A polychoric correlation coefficient for x and y.

    """
    if x_ints is None:
        x_ints = estimate_intervals(x)
    if y_ints is None:
        y_ints = estimate_intervals(y)
    x_ints, x_inds = x_ints
    y_ints, y_inds = y_ints
    p, m = len(x_ints) - 1, len(y_ints) - 1
    n = np.zeros((p, m))
    for a, b in zip(x_inds, y_inds):
        if not (np.isnan(a) or np.isnan(b)):
            n[a - 1, b - 1] += 1
    def calc_likelihood(r):
        return -sum(np.log(bivariate_cdf([x_ints[i], y_ints[j]],
                                 [x_ints[i + 1], y_ints[j + 1]], r)) * n[i, j]
                    for i in range(p) for j in range(m))
    return minimize_scalar(calc_likelihood, bounds=(-1, 1), method='bounded').x
                

def hetcor(data, ords=None, nearest=False):
    """
    Compute a heterogenous correlation matrix.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        DESCRIPTION.
    ords : list, optional
        Names of ordinal variables if data is DataFrame or indices of
        ordinal numbers if data is np.array. If ords are None then ordinal
        variables will be determined automatically. The default is None.
    nearest : bool, optional
        If True, then nearest PD correlation matrix is returned instead. The
        default is False.

    Returns
    -------
    cor : pd.DataFrame
        A heterogenous correlation matrix.

    """
    if type(data) is np.ndarray:
        cov = cor(data)
        if ords is None:
            ords = set()
            for i in range(data.shape[1]):
                if len(np.unique(data[:, i])) / data.shape[0] < 0.3:
                    ords.add(i)
        conts = set(range(data.shape[1])) - set(ords)
    else:
        cov = data.corr()
        if ords is None:
            ords = set()
            for var in data:
                if len(data[var].unique()) / len(data[var]) < 0.3:
                    ords.add(var)
        conts = set(data.columns) - set(ords)
    data = data.T
    c_means = {v: np.nanmean(data[v]) for v in conts}
    c_vars = {v: np.nanvar(data[v]) for v in conts}
    c_z = {v: (data[v] - c_means[v]) / c_vars[v] for v in conts}
    c_pdfs = {v: norm.logpdf(data[v], c_means[v], c_vars[v]) for v in conts}
    o_ints = {v: estimate_intervals(data[v]) for v in ords}

    for c, o in product(conts, ords):
        cov[c][o] = polyserial_corr(data[c], data[o], x_mean=c_means[c],
                                    x_var=c_vars[c], x_z=c_z[c],
                                    x_pdfs=c_pdfs[c], y_ints=o_ints[o])
        cov[o][c] = cov[c][o]
    for a, b in combinations(ords, 2):
        cov[a][b] = polychoric_corr(data[a], data[b], o_ints[a], o_ints[b])
        cov[b][a] = cov[a][b]
    if nearest:
        if type(cov) is pd.DataFrame:
            names = cov.columns
            cov = corr_nearest(cov,threshold=0.05)
            cov = pd.DataFrame(cov, columns=names, index=names)
        else:
            cov = corr_nearest(cov, threshold=0.05)
    return cov
