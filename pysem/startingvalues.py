#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains functions for stating values estimation."""
from scipy.stats import linregress
import numpy as np


def start_beta(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    Parameters in beta are traditionally set to 0 at start.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    return 0.0


def start_lambda(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    Manifest variables are regressed onto their counterpart with fixed
    regression coefficient.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    if rval not in model.vars['latent']:
        return 0.0
    obs = model.vars['observed']
    first = rval
    while first not in obs:
        try:
            first = model.first_manifs[first]
            if first == rval:
                return 0.0
        except KeyError:
            return 0.0
    if first is None or not hasattr(model, 'mx_data'):
        return 0.0
    i, j = obs.index(first), obs.index(lval)
    data = model.mx_data
    x, y = data[:, i], data[:, j]
    mask = np.isfinite(x) & np.isfinite(y)
    return linregress(x[mask], y[mask]).slope


def start_psi(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    Exogenous covariances are fixed to their empirical values.
    All other variances are halved. Latent variances are set to 0.05,
    everything else is set to zero.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    lat = model.vars['latent']
    if rval in lat or lval in lat:
        if rval == lval:
            return 0.05
        return 0.0
    exo = model.vars['exogenous']
    obs = model.vars['observed']
    i, j = obs.index(lval), obs.index(rval)
    if lval in exo:
        return model.mx_cov[i, j]
    elif i == j:
        return model.mx_cov[i, j] / 2
    return 0.0


def start_theta(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    Variances are set to half of observed variances.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    if lval != rval:
        return 0.0
    obs = model.vars['observed']
    i, j = obs.index(lval), obs.index(rval)
    return model.mx_cov[i, j] / 2


def start_gamma1(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    Parameters in Gamma1 are set to 0 at start unless we are dealing with
    means, then they are estimated as means.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    if rval == '1':
        mx = model.mx_data
        i = model.vars['observed'].index(lval)
        return np.nanmean(mx[:, i]) / 2
    return 0.0


def start_gamma2(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    Parameters in Gamma2 are set to 0 at start unless we are dealing with
    means, then they are estimated as means.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    if rval == '1':
        mx = model.mx_data
        i = model.vars['observed'].index(lval)
        return np.nanmean(mx[:, i]) / 2
    return 0.0


def start_d(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    In future a sophisticated procedure will be provided.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    if lval == rval:
        try:
            v = model.effects_loadings.get(lval, 0.1) / 2
        except (AttributeError, TypeError):
            v = 0.05
        return v
    return 0.0


def start_v(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    return 1.0


'''---------------------------------IMPUTER---------------------------------'''


def start_data_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just calculates mean.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    obs = model.mod.vars['observed']
    try:
        i = obs.index(rval)
    except ValueError:
        return 0.0
    mx = model.mod.mx_data
    return np.nanmean(mx[:, i])


def start_g_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just calculates mean.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    obs = model.mod.vars['observed_exogenous']
    try:
        i = obs.index(rval)
    except ValueError:
        return 0.0
    mx = model.mod.mx_g1
    return np.nanmean(mx[i, :])


def start_beta_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just copies values from original model.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    mx = model.mod.mx_beta
    rows, cols = model.mod.names_beta
    i, j = rows.index(lval), cols.index(rval)
    v = mx[i, j]
    return v


def start_gamma1_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just copies values from original model.
    Parameters
    ----------
    model : Model
        Imputer instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    mx = model.mod.mx_gamma1
    rows, cols = model.mod.names_gamma1
    i, j = rows.index(lval), cols.index(rval)
    v = mx[i, j]
    return v


def start_gamma2_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just copies values from original model.
    Parameters
    ----------
    model : Model
        Imputer instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    mx = model.mod.mx_gamma2
    rows, cols = model.mod.names_gamma2
    i, j = rows.index(lval), cols.index(rval)
    v = mx[i, j]
    return v


def start_lambda_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just copies values from original data.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    mx = model.mod.mx_lambda
    rows, cols = model.mod.names_lambda
    i, j = rows.index(lval), cols.index(rval)
    v = mx[i, j]
    return v


def start_psi_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just copies values from original data.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    mx = model.mod.mx_psi
    rows, cols = model.mod.names_psi
    i, j = rows.index(lval), cols.index(rval)
    v = mx[i, j]
    return v


def start_theta_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just copies values from original data.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    mx = model.mod.mx_theta
    rows, cols = model.mod.names_theta
    i, j = rows.index(lval), cols.index(rval)
    v = mx[i, j]
    return v


def start_d_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just copies values from original data.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    mx = model.mod.mx_d
    rows, cols = model.mod.names_d
    i, j = rows.index(lval), cols.index(rval)
    v = mx[i, j]
    return v


def start_v_imp(model, lval: str, rval: str):
    """
    Calculate starting value for parameter in data given data in model.

    For Imputer -- just copies values from original data.
    Parameters
    ----------
    model : Model
        Model instance.
    lval : str
        L-value name.
    rval : str
        R-value name.

    Returns
    -------
    float
        Starting value.

    """
    mx = model.mod.mx_v
    rows, cols = model.mod.names_v
    i, j = rows.index(lval), cols.index(rval)
    v = mx[i, j]
    return v
