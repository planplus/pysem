#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple univariate BLUP implementation for starting values estimation."""
import numpy as np
from scipy.optimize import minimize


def grad(sigmas: np.ndarray, y: np.ndarray, k: np.ndarray):
    v = 1 / (sigmas[0] + sigmas[1] * k)
    if np.any(v < 1e-12):
        return [np.nan, np.nan]
    yt = y * v
    g = np.zeros(2)
    g[0] = np.sum(yt ** 2) - np.sum(np.log(v))
    g[1] = np.sum(yt * k * y) - np.sum(np.log(v ** 2 * k))
    return g


def obj(sigmas: np.ndarray, y: np.ndarray, k: np.ndarray):
    v = 1 / (sigmas[0] + sigmas[1] * k)
    if np.any(v < 1e-8):
        return np.nan
    yt = y * v
    return np.sum(yt * y) - np.sum(np.log(v))


def blup(y: np.ndarray, k: np.ndarray, p=0.8, maxiter=50):
    """
    Calculate BLUP estimate for U of a single variable.

    Parameters
    ----------
    y : np.ndarray
        Observations of a given variable.
    k : np.ndarray
        K matrix.
    p : float, optional
        Expected ratio of variable variance to random effect variance. Used for
        starting values only. The default is 0.8.
    maxiter : int, optional
        Maximal number of iterations. Better not be too high or estimation
        process could take noticable time in some cases. The default is 50.

    Returns
    -------
    U
        Random effects estimate (BLUP).

    """

    v = np.var(y)
    x0 = np.array([p * v, (1 - p) * v])
    s = minimize(lambda x: obj(x, y, k), x0, jac=lambda x: grad(x, y, k),
                 method="SLSQP", options={'maxiter': maxiter},
                 bounds=([0, None], [0, None])
                 ).x
    v = 1 / (1 / s[0] + (1 / s[1]) * (1 / k))
    return y * v / s[0], s
