#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graphviz wrapper to visualie SEM models."""
from .model import Model
import logging
try:
    import graphviz
    __GRAPHVIZ = True
except ModuleNotFoundError:
    logging.info("No graphviz package found, visualization method is "
                 "unavailable")
    __GRAPHVIZ = False
    

def semplot(mod: Model, filename: str, inspection=None, plot_covs=False,
            plot_exos=True, images=None, engine='dot', latshape='circle',
            plot_ests=True, std_ests=False, show=False):
    """
    Draw a SEM diagram.

    Parameters
    ----------
    mod : Model
        Model instance.
    filename : str
        Name of file where to plot is saved.
    inspection : pd.DataFrame, optional
        Parameter estimates as returned by Model.inspect(). The default is
        None.
    plot_covs : bool, optional
        If True, covariances are also drawn. The default is False.
    plot_exos: bool, optional
        If False, exogenous variables are not plotted. It might be useful,
        for example, in GWAS setting, where a number of exogenous variables,
        i.e. genetic markers, is oblivious. Has effect only with ModelMeans or
        ModelEffects. The default is True.
    images : dict, optional
        Node labels can be replaced with images. It will be the case if a map
        variable_name->path_to_image is provided. The default is None.
    engine : str, optional
        Graphviz engine name to use. The default is 'dot'.
    latshape : str, optional
        Graphviz-compaitable shape for latent variables. The default is
        'circle'.
    plot_ests : bool, optional
        If True, then estimates are also plotted on the graph. The default is
        True.
    std_ests : bool, optional
        If True and plot_ests is True, then standardized values are plotted
        instead. The default is False.
    show : bool, optional
        If True, the 

    Returns
    -------
    Graphviz graph.

    """
    if not __GRAPHVIZ:
        raise ModuleNotFoundError("No graphviz module is installed.")
    if type(mod) is str:
        mod = Model(mod)
    if not hasattr(mod, 'last_result'):
        plot_ests = False
    if inspection is None:
        inspection = mod.inspect(std_est=std_ests)
    if images is None:
        images = dict()
    if std_ests:
        inspection['Estimate'] = inspection['Est. Std']
    t = filename.split('.')
    filename, ext = '.'.join(t[:-1]), t[-1]
    g = graphviz.Digraph('G', format=ext, engine=engine)
    
    g.attr(overlap='scale', splines='true')
    g.attr('edge', fontsize='12')
    g.attr('node', shape=latshape, fillcolor='#cae6df', style='filled')
    for lat in mod.vars['latent']:
        if lat in images:
            g.node(lat, label='', image=images[lat])
        else:
            g.node(lat, label=lat)
    
    g.attr('node', shape='box', style='')
    for obs in mod.vars['observed']:
        if obs in images:
            g.node(obs, label='', image=images[obs])
        else:
            g.node(obs, label=obs)

    regr = inspection[inspection['op'] == '~']
    all_vars = mod.vars['all']
    try:
        exo_vars = mod.vars['observed_exogenous']
    except KeyError:
        exo_vars = set()
    for _, row in regr.iterrows():
        lval, rval, est = row['lval'], row['rval'], row['Estimate']
        if (rval not in all_vars) or (~plot_exos and rval in exo_vars) or\
            (rval == '1'):
            continue
        if plot_ests:
            pval = row['p-value']
            label = '{:.3f}'.format(float(est))
            if pval !='-':
                label += r'\np-val: {:.2f}'.format(float(pval))
        else:
            label = str()
        g.edge(rval, lval, label=label)
    if plot_covs:
        covs = inspection[inspection['op'] == '~~']
        for _, row in covs.iterrows():
            lval, rval, est = row['lval'], row['rval'], row['Estimate']
            if lval == rval:
                continue
            if plot_ests:
                pval = row['p-value']
                label = '{:.3f}'.format(float(est))
                if pval !='-':
                    label += r'\np-val: {:.2f}'.format(float(pval))
            else:
                label = str()
            g.edge(rval, lval, label=label, dir='both', style='dashed')
    g.render(filename, view=show)
    return g