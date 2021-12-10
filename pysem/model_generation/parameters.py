# -*- coding: utf-8 -*-
"""The module is responsible for generating parameters given model
description."""
from .description import generate_desc
from semopy import ModelMeans
import pandas as pd
import random


def param_variance():
    return random.uniform(0.7, 1.3)

def param_correlation():
    return random.uniform(-0.3, 0.3)

def param_loading():
    return random.choice([1, -1]) * random.uniform(0.3, 1.5)




random.seed(123)

desc = generate_desc(3, 2, 3, 3, 0)

def generate_parameters(desc: str, intercepts=False,
                        sampler_var_psi=param_variance,
                        sampler_cor_psi=param_correlation,
                        sampler_var_theta=param_variance,
                        sampler_cor_theta=param_correlation,
                        sampler_reg_beta=param_loading,
                        sampler_reg_lambda=param_loading,
                        sampler_reg_gamma=param_loading):
    """
    Generate random parameters for a given model.

    Parameters
    ----------
    desc : str
        Description of semopy model.
    intercepts : bool, optional
        Should intercepts be included in the model? The default is False.
    sampler_var_psi : callable, optional
        Method that samples variance for Psi matrix. The default is
        param_variance.
    sampler_cor_psi : callable, optional
        Method that samples correlation for Psi matrix.. The default is
        param_correlation.
    sampler_var_theta : callable, optional
        Method that samples variance for Theta matrix. The default is
        param_variance.
    sampler_cor_theta : callable, optional
        Method that samples correlation for Theta matrix. The default is
        param_correlation.
    sampler_reg_beta : callable, optional
        Method that samples loading for Beta matrix. The default is
        param_loading.
    sampler_reg_lambda : callable, optional
        Method that samples loading for Lambda matrix. The default is
        param_loading.
    sampler_reg_gamma : callable, optional
        Method that samples loading for Gamma matrix. The default is
        param_loading.

    Returns
    -------
    Pandas DataFrame with parameters values, dict of matrices, filled with
    parameters.
    """
    d = {'lval': [], 'op': [], 'rval': [], 'Estimate': []}
    m = ModelMeans(desc, intercepts=intercepts)
    matrices = m.inspect('mx', what='params')
    
    psi_rows = matrices['Psi'].index
    psi = matrices['Psi'].values
    for i in range(psi.shape[0]):
        if type(psi[i, i]) is str:
            p = m.parameters[psi[i, i]]
            if p.active:
                psi[i, i] = sampler_var_psi()
                d['lval'].append(psi_rows[i])
                d['op'].append('~~')
                d['rval'].append(psi_rows[i])
                d['Estimate'].append(psi[i, i])
            else:
                psi[i, i] = p.start
    for i in range(psi.shape[0]):
        for j in range(psi.shape[1]):
            if i != j and type(psi[i, j]) is str:
                p = m.parameters[psi[i, j]]
                if p.active:
                    t = (psi[i, i] * psi[j, j]) ** (0.5)
                    psi[i, j] = sampler_cor_psi() * t
                    psi[j, i] = psi[i, j]
                    d['lval'].append(psi_rows[i])
                    d['op'].append('~~')
                    d['rval'].append(psi_rows[j])
                    d['Estimate'].append(psi[i, j])
                else:
                    psi[i, j] = p.start
                    psi[j, i] = p.start
    m.mx_psi = psi.astype('float64')
    
    theta_rows = matrices['Theta'].index
    theta = matrices['Theta'].values
    for i in range(theta.shape[0]):
        if type(theta[i, i]) is str:
            p = m.parameters[theta[i, i]]
            if p.active:
                theta[i, i] = sampler_var_theta()
                d['lval'].append(theta_rows[i])
                d['op'].append('~~')
                d['rval'].append(theta_rows[i])
                d['Estimate'].append(theta[i, i])
            else:
                theta[i, i] = p.start
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if i != j and type(theta[i, j]) is str:
                p = m.parameters[theta[i, j]]
                if p.active:
                    t = (theta[i, i] * theta[j, j]) ** (0.5)
                    theta[i, j] = sampler_cor_theta() * t
                    theta[j, i] = theta[i, j]
                    d['lval'].append(theta_rows[i])
                    d['op'].append('~~')
                    d['rval'].append(theta_rows[j])
                    d['Estimate'].append(theta[i, j])
                else:
                    theta[i, j] = p.start
                    theta[j, i] = p.start
    m.mx_theta = theta.astype('float64')
    
    beta_rows = matrices['Beta'].index
    beta_cols = matrices['Beta'].columns
    beta = matrices['Beta'].values
    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            if type(beta[i, j]) is str:
                p = m.parameters[beta[i, j]]
                if p.active:
                    beta[i, j] = sampler_reg_beta()
                    d['lval'].append(beta_rows[i])
                    d['op'].append('~')
                    d['rval'].append(beta_cols[j])
                    d['Estimate'].append(beta[i, j])
                else:
                    beta[i, j] = p.start
    m.mx_beta = beta.astype('float64')

    lamb_rows = matrices['Lambda'].index
    lamb_cols = matrices['Lambda'].columns
    lamb = matrices['Lambda'].values
    s = set()
    firsts = m.first_manifs
    for j in range(lamb.shape[1]):
        for i in range(lamb.shape[0]):
            if type(lamb[i, j]) is str:
                p = m.parameters[lamb[i, j]]
                if p.active:
                    if firsts.get(lamb_cols[j], None) == lamb_rows[i]:
                        lamb[i, j] = 1.0
                        s.add(lamb_cols[j])
                    else:
                        lamb[i, j] = sampler_reg_lambda()
                    d['lval'].append(lamb_rows[i])
                    d['op'].append('~')
                    d['rval'].append(lamb_cols[j])
                    d['Estimate'].append(lamb[i, j])
                else:
                    lamb[i, j] = p.start
    m.mx_lambda = lamb.astype('float64')

    gamma_rows = matrices['Gamma1'].index
    gamma_cols = matrices['Gamma1'].columns
    gamma = matrices['Gamma1'].values
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            if type(gamma[i, j]) is str:
                p = m.parameters[gamma[i, j]]
                if p.active:
                    gamma[i, j] = sampler_reg_gamma()
                    d['lval'].append(gamma_rows[i])
                    d['op'].append('~')
                    d['rval'].append(gamma_cols[j])
                    d['Estimate'].append(gamma[i, j])
                else:
                    gamma[i, j] = p.start
    m.mx_gamma1 = gamma.astype('float64')

    gamma_rows = matrices['Gamma2'].index
    gamma_cols = matrices['Gamma2'].columns
    gamma = matrices['Gamma2'].values
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            if type(gamma[i, j]) is str:
                p = m.parameters[gamma[i, j]]
                if p.active:
                    gamma[i, j] = sampler_reg_gamma()
                    d['lval'].append(gamma_rows[i])
                    d['op'].append('~')
                    d['rval'].append(gamma_cols[j])
                    d['Estimate'].append(gamma[i, j])
                else:
                    gamma[i, j] = p.start
    m.mx_gamma2 = gamma.astype('float64')

    return pd.DataFrame.from_dict(d), m