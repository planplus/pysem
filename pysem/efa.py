#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explarotary Factor Analysis (EFA).

Methods that help to retrieve latent structure CFA-style. Here only our own
unorthodox approach is considered: we determine number of latent factors and
their loadings via clustering analysis.
"""
import numpy as np
import pandas as pd
from .model import Model
from collections import defaultdict
from sklearn.cluster import OPTICS
from sklearn.decomposition import SparsePCA
from itertools import permutations
from copy import deepcopy
from .utils import cor


def find_latents(data: pd.DataFrame, min_loadings=2, mx_cor=None,
                 underscript='', mode='spca', spca_alpha=1):
    """
    Retrieve number of latent factors and their simple loadings.

    Simple loadings here mean the simplest model where none of latent factors
    have joint indicator. Furthermore, this methods performs no SEM-based
    verification of the proposed CFA-model.
    Parameters
    ----------
    data : pd.DataFrame
        Data.
    min_loadings : int, optional
        Minimal number of indicators per latent factor. The default is 2.
    mx_cor : pd.DataFrame, optional
        Correlation matrix will be used instead of data if provided. The
        default is None.
    underscript : str, optional
        Underscript to add after latent factor names. The default is ''.
    mode : str, optional
        If 'optics', then a heuristics appraoch based entirely on OPTICS is
        used. If "spca", then OPTICS is used only to find number of latent
        factors, and loadings are found using sprase PCA. The default is 
        'optics'.
    spca_alpha : float, optional
        If mode == 'spca', then it is a regularization multiplier. The default
        is 1.0.
    Returns
    -------
    tuple
        Mapping latent factor->indicators and distance matrix.

    """
    if mx_cor is None:
        names = data.columns
        mx_cor = cor(data)
    else:
        names = mx_cor.columns
        mx_cor = mx_cor.values
    names = list(names)
    dist = np.clip(1.0 - np.abs(mx_cor), 0.0, 1.0)
    clust = OPTICS(min_samples=min_loadings, metric='precomputed').fit(dist)
    loadings = defaultdict(set)
    if mode == 'optics':
        for i, label in enumerate(clust.labels_):
            if label != -1:
                name = f'eta{label+1}{underscript}'
            else:
                name = -1
            loadings[name].add(names[i])
    else:
        num_lats = set(clust.labels_)
        num_lats = len(num_lats - {-1})
        if data is None:
            raise Exception("For sparse PCA mode data must be provided.")
        cmp = SparsePCA(num_lats, alpha=spca_alpha).fit(data).components_
        for i in range(cmp.shape[0]):
            inds = np.where(abs(cmp[i]) >= 0.05)[0]
            inds = data.columns[inds]
            name = f'eta{i+1}{underscript}'
            loadings[name] = set(inds)
    for lat, inds in loadings.items():
        if lat == -1:
            continue
        dists = list()
        inds = list(inds)
        for ind in inds:
            d = 0
            i = names.index(ind)
            for ind0 in inds:
                if ind0 != ind:
                    j = names.index(ind0)
                    d += dist[i, j]
            dists.append(d)
        inds = sorted(inds, key=lambda x: dists[inds.index(x)])
        loadings[lat] = inds
    return loadings, pd.DataFrame(dist, columns=names, index=names)


def dict_to_desc(d: dict):
    """
    Transform dictionary into a text-based description.

    Parameters
    ----------
    d : dict
        Mapping factors->indicators.

    Returns
    -------
    desc : str
        Text-form description of the model.

    """
    desc = str()
    for lat, inds in d.items():
        if lat != -1:
            inds = ' + '.join(inds)
            desc += f'{lat} =~ {inds}\n'
    return desc


def finalize_loadings(loadings: dict, data: pd.DataFrame, dist: pd.DataFrame,
                      pval=0.01, model=Model, base_desc='', only_clean=False):
    """
    Test p-values of CFA-like model and find cross-loadings between factors.

    Parameters
    ----------
    loadings : dict
        Mapping factor->indicators.
    data : pd.DataFrame
        Dataset.
    dist : pd.DataFrame
        Distance matrix.
    pval : float, optional
        Statistical significiance cut-off value. The default is 0.01.
    model : TYPE, optional
        Class instance. The default is Model.
    base_desc : str, optional
        Text description to append to each model. The default is ''.
    only_clean : bool, optional
        If True, then only high p-value loadings are dropped, no
        intercorrelations are studied. The default is False.

    Returns
    -------
    dict
        Mapping factor->indicators.

    """

    def clean_loadings(loadings: dict, base_desc: str):
        desc = dict_to_desc(loadings)
        if base_desc:
            desc += '# Base desc:\n' + base_desc
        m = model(desc)
        m.fit(data)
        ins = m.inspect()
        lats_to_remove = set()
        for lat, inds in loadings.items():
            if lat != -1:
                inds = iter(inds)
                # if not base_desc:
                next(inds)
                to_remove = set()
                for ind in inds:
                    if not is_loading_significant(ins, ind, lat, pval=pval):
                        to_remove.add(ind)
                        if -1 in loadings:
                            loadings[-1].add(ind)
                it = loadings[lat]
                list(map(lambda x: it.remove(x), to_remove))
                if len(it) < 2:
                    lats_to_remove.add(lat)
        for lat in lats_to_remove:
            del loadings[lat]
        if -1 in loadings:
            for lat, inds in list(loadings.items()):
                if len(inds) == 1 and lat != -1:
                    del loadings[lat]
                    loadings[-1].append(inds[0])

    def test(loadings: dict, base_desc: str, lat: str, ind: str):
        desc = dict_to_desc(loadings)
        desc += base_desc
        desc += f'{lat} =~ {ind}'
        m = model(desc)
        if not m.fit(data).success:
            return float('inf')
        ins = m.inspect()
        if m._fim_warn:
            return float('inf')
        return get_loading_significiance(ins, ind, lat)
    loadings = deepcopy(loadings)
    clean_loadings(loadings, base_desc)
    if only_clean:
        return loadings
    loadings_comp = deepcopy(loadings)
    try:
        del loadings_comp[-1]
    except KeyError:
        pass
    lats = [lat for lat in loadings if lat != -1]
    for ind in loadings[-1]:
        pvals = [test(loadings, base_desc, lat, ind) for lat in lats]
        i = np.argmin(pvals)
        if pvals[i] < pval:
            loadings_comp[lats[i]].append(ind)
    loadings = loadings_comp
    clean_loadings(loadings, base_desc)
    loadings_joint = deepcopy(loadings)
    for a, b in permutations(loadings, 2):
        b_items = loadings[b]
        a_items = sorted(loadings[a],
                         key=lambda x: min(dist.loc[x, b_items]))
        for item in a_items:
            kt = test(loadings_joint, base_desc, b, item)
            if kt < pval:
                loadings_joint[b].append(item)
            else:
                # continue
                break
    tmp = None
    while loadings_joint != tmp:
        tmp = deepcopy(loadings_joint)
        clean_loadings(loadings_joint, base_desc)
    return loadings_joint


def explore_cfa_model(data: pd.DataFrame, min_loadings=2, pval=0.01,
                      model=Model, ret_desc=True, mode='spca',
                      spca_alpha=1.0):
    """
    Retrieve CFA model from data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset.
    min_loadings : int, optional
        Minimal number of indicators per factor. The default is 2.
    pval : TYPE, optional
        P-value cutoff. The default is 0.01.
    model : TYPE, optional
        Class to instantiate. The default is Model.
    ret_desc : TYPE, optional
        If True, text-based description is returned. If False, a mapping is
        returned instead. The default is True.
    mode : str, optional
        If 'optics', then a heuristics appraoch based entirely on OPTICS is
        used. If "spca", then OPTICS is used only to find number of latent
        factors, and loadings are found using sprase PCA. The default is 
        'optics'.
    spca_alpha : float, optional
        If mode == 'spca', then it is a regularization multiplier. The default
        is 1.0.

    Returns
    -------
    str
        Model description.

    """
    loadings, dist = find_latents(data, min_loadings=min_loadings, mode=mode,
                                  spca_alpha=spca_alpha)
    loadings = finalize_loadings(loadings, data=data, dist=dist, pval=pval,
                                 model=Model, only_clean=mode == 'spca')
    return dict_to_desc(loadings) if ret_desc else loadings


def explore_pine_model(data: pd.DataFrame, min_loadings=2, pval=0.01, levels=2,
                       model=Model):
    """
    Retrieve pine-like model from the data.

    "Pine-like" means that we also assume a layer of latent factors above the
    CFA model that regresses onto latent factors in the CFA model.
    Parameters
    ----------
    data : pd.DataFrame
        Dataset.
    min_loadings : int, optional
        Minimal number of loadings per factor. The default is 2.
    pval : float, optional
        P-value cutoff. The default is 0.01.
    levels : int, optional
        Maximal number of layers. The default is 2.
    model : Model, optional
        Class to instantiate. The default is Model.

    Returns
    -------
    str
        Model description.

    """
    cfa = explore_cfa_model(data, min_loadings=min_loadings, pval=pval,
                            model=model, ret_desc=False)
    to_remove = {lat for lat, items in cfa.items() if len(items) < 2}
    for lat in to_remove:
        del cfa[lat]
    pine = dict_to_desc(cfa)
    names = list(cfa.keys())
    min_loadings = 2
    for level in range(1, levels):
        m = model(pine)
        m.fit(data)
        psi = m.inspect('mx')['Psi'].loc[names, names]
        d = np.diag(psi.values.diagonal() ** (-0.5))
        corr = d @ psi @ d
        corr.columns = names
        corr.index = names
        lats, dist = find_latents(None, mx_cor=corr, min_loadings=2,
                                  underscript=f'_{level}')
        joint = finalize_loadings(lats, data, dist, pval=pval, model=model,
                                  base_desc=pine)
        if joint:
            pine += 'DEFINE(latent) {}\n'.format(' '.join(joint.keys()))
        else:
            break
        names = list(joint.keys())
        pine += dict_to_desc(joint)
    return pine.strip()


def get_loading_significiance(res, lval: str, rval: str):
    """
    Get p-value of certain regression coefficient.

    Parameters
    ----------
    res : Model or pd.DataFrame
        Either fitted Model or pandas DataFrame.
    lval : str
        Left-value of operation.
    rval : str
        Right-value of operation.

    Returns
    -------
    float
        p-value.

    """
    if type(res) is not pd.DataFrame:
        res = res.inspect()
    inds = (res.lval == lval) & (res.rval == rval) & (res.op == '~')
    p = res[inds]['p-value'].values[0]
    return p


def is_loading_significant(res, lval: str, rval: str, pval=0.01):
    """
    Test significiance of a regression coefficient.

    Parameters
    ----------
    res : Model or pd.DataFrame
        Either fitted Model or pandas DataFrame.
    lval : str
        Left-value of operation.
    rval : str
        Right-value of operation.
    pval : TYPE, optional
        p-value cutoff. The default is 0.01.

    Returns
    -------
    bool
        True if significiant, False otherwise.
    """
    p = get_loading_significiance(res, lval, rval)
    if p < pval:
        return True
    return False
