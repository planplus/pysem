#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regularization module."""
from functools import partial
import numpy as np


def create_regularization(model, regularization='l1-thresh', c=1.0, alpha=None,
                          param_names=None, mx_names=None):
    """
    Build additive regularization terms for loss funtions of SEM models.

    Supported regularizators:
        'l1-naive':  Straightforward L1 regularization (unrecommended).
        'l1-smooth': Smooth approximization of L1 regularization.
        'l1-thresh': L1 regularization with a soft thresholding operator.
        'l2-naive':  Straightforward L2 regularization
        'l2-square': L2 regularization squared.

    Parameters
    ----------
    model : Model
        Model reference.
    regularization : str, optional
        Type of of regularization to use. The default is 'l1-thresh'.
    c : TYPE, optional
        Regularization constant. The default is 1.0.
    alpha : float, optional
        If regularization is "l1-smooth", it's the alpha multiplier: the higher
        it's value is, the more accurate is the approximation. If
        regularization is "l1-thresh", then its a soft thresholding operator
        parameter: the closer to zero it is, the more accurate is the operator.
        The default is 1e-6.
    param_names : iterable, optional
        Parameter names to regularize. The default is None.
    mx_names : iterable, optional
        Parameters that are in mx_names will be regularized. The default is
        None.

    Returns
    -------
    Pair of objective and, if possible, a gradient function to be passed into
    Model.

    """
    inds = list()
    if param_names is None:
        param_names = set()
    if mx_names is None:
        mx_names = set()
    mx_names = [id(getattr(model, 'mx_{}'.format(mx.lower())))
                for mx in mx_names]
    i = 0
    for name, p in model.parameters.items():
        if not p.active:
            continue
        if name in param_names:
            inds.append(i)
        else:
            t = False
            for loc in p.locations:
                try:
                    if id(loc.matrix) in mx_names:
                        t = True
                        break
                except ValueError:
                    pass
            if t:
                inds.append(i)
        i += 1
    inds = np.array(inds, dtype=np.int)
    if regularization == 'l1-naive':
        obj = partial(l1_naive, c=c, inds=inds)
        grad = partial(l1_naive_grad, c=c, inds=inds)
    elif regularization == 'l1-smooth':
        if alpha is None:
            alpha = 1e6
        else:
            alpha = 1 / alpha
        obj = partial(l1_smooth, c=c, alpha=alpha, inds=inds)
        grad = partial(l1_smooth_grad, c=c, alpha=alpha, inds=inds)
    elif regularization == 'l1-thresh':
        if alpha is None:
            alpha = 1e-6
        obj = partial(l1_thresh, c=c, alpha=alpha, inds=inds)
        grad = partial(l1_thresh_grad, c=c, alpha=alpha, inds=inds)
    elif regularization == 'l2-naive':
        obj = partial(l2_naive, c=c, inds=inds)
        grad = partial(l2_naive_grad, c=c, inds=inds)
    elif regularization == 'l2-square':
        obj = partial(l2_square, c=c, inds=inds)
        grad = partial(l2_square_grad, c=c, inds=inds)
    else:
        raise NotImplementedError(f"Unknown regularization {regularization}.")
    return obj, grad


def l1_naive(x: np.ndarray, c: float, inds: np.ndarray):
    return c * np.abs(x[inds]).sum()


def l1_naive_grad(x: np.ndarray, c: float, inds: np.ndarray):
    g = np.zeros(len(x))
    for i in inds:
        g[i] = c * np.sign(x[i])
    return g


def l1_smooth(x: np.ndarray, c: float, alpha: float, inds: np.ndarray):
    t = alpha * x[inds]
    e_m = np.exp(-t)
    e_p = np.exp(t)
    r= (np.log(e_p + 1) + np.log(e_m + 1)).sum() / alpha
    return r


def l1_smooth_grad(x: np.ndarray, c: float, alpha: float, inds: np.ndarray):
    g = np.zeros(len(x))
    g[inds] = x[inds]
    g = np.tanh(alpha / 2 * g) 
    return g


def l1_thresh(x: np.ndarray, c: float, alpha: float, inds: np.ndarray):
    return c * np.abs(x[inds]).sum()


def l1_thresh_grad(x: np.ndarray, c: float, alpha: float, inds: np.ndarray):
    g = np.zeros(len(x))
    g[inds] = x[inds]
    t = g > alpha
    g[t] -= alpha
    t = g < -alpha
    g[t] += alpha
    t = np.abs(g) < alpha
    g[t] = 0.0
    return g


def l2_naive(x: np.ndarray, c: float, inds: np.ndarray):
    return c * np.sqrt(np.square(x[inds]).sum())


def l2_naive_grad(x: np.ndarray, c: float, inds: np.ndarray):
    g = np.zeros(len(x))
    t = x[inds]
    sqr = np.sqrt(np.square(t).sum())
    for i in inds:
        g[i] = x[i] / sqr
    return c * g


def l2_square(x: np.ndarray, c: float, inds: np.ndarray):
    return c * np.square(x[inds]).sum()


def l2_square_grad(x: np.ndarray, c: float, inds: np.ndarray):
    g = np.zeros(len(x))
    for i in inds:
        g[i] = 2 * c * x[i]
    return g
