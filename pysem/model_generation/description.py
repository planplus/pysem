# -*- coding: utf-8 -*-
"""The module is responsible for generating model descriptions/specifications
given some parameters."""
from copy import deepcopy
import random


def generate_cfa(n_lat: int, n_inds: int, p_join=0.0, max_join=1,
                 base_lat='eta', base_ind='y'):
    """
    Generate random CFA model.

    Parameters
    ----------
    n_lat : int, tuple
        Number of latent factors in the model. If tuple, then the number of
        factors is chosen randomly from the interval.
    n_inds : int, tuple
        Number of indicators per factor in the model. If tuple, then the number
        of indicators is chosen randomly from the interval for each factor.
    p_join : float
        A probability that the indicator will be shared with another latent
        factor. Closely related to max_join, as the test whether to add the
        indicator to another random variable is evaluated for each other latent
        factor until either there is no factors left or until it has beed added
        to max_join factors. The default is 0.0.
    max_join : int
        Maximal number of factors per indicator.
    base_lat : str, tuple
        Base of a name for latent variable. If tuple, then the tuple should
        be of length n_lat and then it contains names of the variables. The
        default is 'eta'.
    base_ind : str
        Base of a name for indocator variable. The default is 'y'.

    Returns
    -------
    Adjacency dictionary: factor -> indicators.

    """
    res = dict()
    if type(n_lat) in (tuple, list):
        n_lat = random.randint(n_lat[0], n_lat[1])
    ind_count = 0
    for i in range(n_lat):
        m = n_inds
        if type(m) in (tuple, list):
            m = random.randint(m[0], m[1])
        if type(base_lat) in (tuple, list):
            lat = base_lat[i]
        else:
            lat = f'{base_lat}{i + 1}'
        lt = list()
        for _ in range(m):
            ind_count += 1
            ind = f'{base_ind}{ind_count}'
            lt.append(ind)
        res[lat] = lt
    tres = deepcopy(res)
    for lat, inds in tres.items():
        for a in inds:
            for latb in tres:
                if latb != lat and random.uniform(0, 1) < p_join:
                    res[latb].append(a)
    return res


def generate_sem(n_endo: int, n_exo: int, n_cycles: int, p_edge=0.3, cfa=None,
                 strict_exo=True, base_endo='x', base_exo='g'):
    """
    Generate SEM model (CFA + PA).

    Parameters
    ----------
    n_endo : int, tuple
        Number of endogenous variables. If tuple, then the number of
        endogenous variables is chosen randomly from the interval.
    n_exo : int, tuple
        Number of exogenous variables (or integer range). If 0, then actual
        number of exogenous variable will still be 1, but it is in fact will
        be one of the "endogenous" (not anymore!) variables.
    n_cycles : int, tuple
        Number of cycles. If tuple, then the number of
        cycles is chosen randomly from the interval.
    p_edge: float
        Probability that an extra edge (i.e. the one that is not required
        to make graph connected) will be added. The default is 0.3.
    cfa : dict, optional
        CFA part of the model. The default is None.
    strict_exo : bool, optional
        If True, then there will be exactly n_exo exogenous variables. The
        default is True.
    base_endo : str, tuple, optional
        Base of a name for endogenous variable. If tuple, then the tuple should
        be of length n_exo and then it contains names of the variables. The
        default is 'x'.
    base_exo : str, tuple, optional
        Base of a name for exogenous variable. If tuple, then the
        tuple should be of length n_exo and then it contains names of
        the variables. The default is 'g'.

    Returns
    -------
    Adjacency dictionary: rval -> lval.
    (i.e. lval ~ rval)
    """
    if type(n_endo) in (tuple, list):
        n_endo = random.randint(n_endo[0], n_endo[1])
    if type(n_cycles) in (tuple, list):
        n_cycles = random.randint(n_cycles[0], n_cycles[1])
    if type(n_exo) in (tuple, list):
        n_exo = random.randint(n_exo[0], n_exo[1])
    if type(base_exo) in (tuple, list) and len(base_exo) == n_exo:
        exos = base_exo
    else:
        exos = [f'{base_exo}{i + 1}' for i in range(n_exo)]
    if type(base_endo) is str:
        endos = [f'{base_endo}{i + 1}' for i in range(n_endo)]
    else:
        endos = base_endo
    if cfa:
        endos += list(cfa.keys())
        res = deepcopy(cfa)
    else:
        res = dict()
    random.shuffle(endos)
    nodes = exos + endos
    for i in range(len(nodes) - 1):
        sx = max(len(exos), i + 1)
        rv = nodes[i]
        lv = random.choice(nodes[sx:])
        lt = res.get(rv, list())
        lt.append(lv)
        for i in range(sx, len(nodes)):
            if nodes[i] != lv and random.uniform(0, 1) < p_edge:
                lt.append(nodes[i])
        res[rv] = lt

    if strict_exo:
        endogenous = set()
        for _, ins in res.items():
            endogenous.update(ins)
        left_out = set(endos) - endogenous
        for v in left_out:
            i = nodes.index(v)
            if i:
                rv = random.choice(nodes[:i])
                if rv not in res:
                    res[rv] = list()
                res[rv].append(v)
    if n_cycles:
        n = len(exos)
        s_nodes = nodes[n:]
        random.shuffle(s_nodes)
        for v in s_nodes:
            i = nodes.index(v)
            if i > n:
                v2 = random.choice(nodes[n:i])
                if v not in res:
                    res[v] = [v2]
                else:
                    res[v].append(v2)
                n_cycles -= 1
                if not n_cycles:
                    break
    return res


def dict_to_desc(d: dict, lats=None):
    """
    Translate dictionary to string that can be used by semopy models.

    Parameters
    ----------
    d : dict
        Dict containing mapping rval->lval (i.e. lval ~ rval).
    lats : dict, optional
        Measurement part. The default is None.

    Returns
    -------
    String that can be fed to semopy model.

    """
    measurement_part = list()
    structural_part = list()
    defines = set()
    d = deepcopy(d)
    to_rem = set()
    if lats:
        for lat, inds in lats.items():
            if lat not in d:
                defines.add(lat)
            else:
                inds = list(filter(lambda x: x not in d, inds))
                if not inds:
                    defines.add(lat)
                    continue
                rvals = ' + '.join(inds)
                s = f'{lat} =~ {rvals}'
                measurement_part.append(s)
                lt = d.get(lat, None)
                if lt:
                    for ind in inds:
                        lt.remove(ind)
                    if not lt:
                        to_rem.add(lat)
    for lat in to_rem:
        del d[lat]
    mappings = dict()
    for rval, lvals in d.items():
        for lval in lvals:
            if lval in mappings:
                lt = mappings[lval]
            else:
                lt = list()
            lt.append(rval)
            mappings[lval] = lt
    for lval, rvals in mappings.items():
        rvals = ' + '.join(rvals)
        s = f'{lval} ~ {rvals}'
        structural_part.append(s)
    res = str()
    if measurement_part:
        res += '# Measurement part:\n' + '\n'.join(measurement_part) + '\n'
    if defines:
        res += '## Latent variables that ought to be defined:\n'
        s = ' '.join(defines)
        res += f'DEFINE(latent) {s}\n'
    if structural_part:
        res += '# Structural part:\n' + '\n'.join(structural_part)
    return res


def generate_desc(n_endo: int, n_exo: int, n_lat: int, n_inds=3,
                  n_cycles=0, p_join=0.0, max_join=1, p_edge=0.3,
                  strict_exo=True, base_lat='eta', base_ind='y', base_endo='x',
                  base_exo='g', ):
    """
    Generate string description of a SEM model, suitable for semopy models.

    Parameters
    ----------
    n_endo : int, tuple
        Number of endogenous variables. If tuple, then the number of
        endogenous variables is chosen randomly from the interval.
    n_exo : int, tuple
        Number of exogenous variables (or integer range). If 0, then actual
        number of exogenous variable will still be 1, but it is in fact will
        be one of the "endogenous" (not anymore!) variables.
    n_lat : int, tuple
        Number of latent factors in the model. If tuple, then the number of
        factors is chosen randomly from the interval.
    n_inds : int, tuple, optional
        Number of indicators per factor in the model. If tuple, then the number
        of indicators is chosen randomly from the interval for each factor.
        The default is 3.
    n_cycles : int, tuple
        Number of cycles. If tuple, then the number of cycles is chosen
        randomly from the interval.
    p_join : float
        A probability that the indicator will be shared with another latent
        factor. Closely related to max_join, as the test whether to add the
        indicator to another random variable is evaluated for each other latent
        factor until either there is no factors left or until it has beed added
        to max_join factors. The default is 0.0.
    max_join : int
        Maximal number of factors per indicator.
    base_lat : str, tuple
        Base of a name for latent variable. If tuple, then the tuple should
        be of length n_lat and then it contains names of the variables. The
        default is 'eta'.
    base_ind : str
        Base of a name for indocator variable. The default is 'y'.

    Returns
    -------
    String description of model, suitable for semopy models.
    """
    cfa = generate_cfa(n_lat, n_inds, p_join=p_join, max_join=max_join,
                       base_lat=base_lat, base_ind=base_ind, )
    sem = generate_sem(n_endo, n_exo, n_cycles, p_edge=p_edge, cfa=cfa,
                       strict_exo=strict_exo, base_endo=base_endo,
                       base_exo=base_exo)
    return dict_to_desc(sem, cfa)
