# -*- coding: utf-8 -*-
"""The module is responsible for generating data for a model.
"""
from .. import ModelMeans
import pandas as pd
import numpy as np
import scipy


def data_exogenous(shape: tuple):
    return np.random.normal(size=shape)


def generate_data(model: ModelMeans, n: int, drop_lats=True,
                  generator_exo=data_exogenous):
    """
    Generate data for a given model with parameters fit.

    Parameters
    ----------
    model : ModelMeans
        ModelMeans that will generate data. It is assumed that the model used
        is just an auxiliarily structure that is returned by other generator
        functions, yet it is fine to use with any custom ModelMeans.
    n : int
        Number of data samples.
    drop_lats : bool, optional
        If True, latent variables are dropped from the dataset. The default is
        True.

    Returns
    -------
    Padnas DataFrame with the generated data.
    """
    if hasattr(model, 'calc_t'):
        sigma, (m, c) = model.calc_sigma()
        mean = model.calc_mean(m)
        t = model.calc_t(sigma)
        l = model.calc_l(sigma)
        tr_l = np.trace(l)
        res = scipy.stats.matrix_normal.rvs(mean, l / tr_l, t)
        res = np.append(res, model.mx_g, axis=0)
        cols = model.vars['observed'] + model.vars['observed_exogenous']
    else:
        c = np.linalg.inv(np.identity(model.mx_beta.shape[0]) - model.mx_beta)
        if c.shape[0]:
            epsilon = np.random.multivariate_normal(mean=np.zeros(c.shape[0]),
                                                    cov=model.mx_psi, size=n)
        else:
            epsilon = None
        lambc = model.mx_lambda @ c
        m = model.mx_lambda.shape[0]
        delta = np.random.multivariate_normal(mean=np.zeros(m),
                                              cov=model.mx_theta, size=n)
        try:
            exo = model.vars['observed_exogenous']
            if generator_exo:
                gamma = generator_exo((len(exo), n))
                try:
                    i = exo.index('1')
                    gamma[i, :] = 1
                except ValueError:
                    pass
            else:
                gamma = model.mx_g
            res = lambc @  model.mx_gamma1 @ gamma + delta.T
            if epsilon is not None:
                res += lambc @ epsilon.T
            res += model.mx_gamma2 @ gamma
            res = np.append(res, gamma, axis=0)
            cols = model.vars['observed'] + exo
        except KeyError:
            res = lambc @ epsilon.T + delta.T
            cols = model.vars['observed']
    df = pd.DataFrame(res.T, columns=cols)
    if not drop_lats and not hasattr(model, 'calc_t') and epsilon is not None:
        omega = c @ epsilon.T
        for i, v in enumerate(model.vars['inner']):
            if v in model.vars['latent']:
                df[v] = omega[i, :]
    if '1' in df.columns:
        df = df.drop('1', axis=1)
    return df
