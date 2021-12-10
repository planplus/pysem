#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-group analysis functionality.
"""

from . import Model
from .stats import calc_stats
from dataclasses import dataclass
import pandas as pd

@dataclass
class MultigroupResult:
    """Structure for printing info on multiple semopy runs, each per group."""

    groups: list
    estimates: dict
    stats: dict
    runs: dict
    n_obs: dict

    def __str__(self):
        s = 'semopy: Multi-Group analysis report\n\n'
        s += 'Groups: {}\n\n'.format(', '.join(self.groups))
        s += 'Number of observations per group:\n'
        s += '\n'.join('{}:\t{}'.format(g, self.n_obs[g]) for g in self.groups)
        s += '\n\nParameter estimates:\n'
        for i, g in enumerate(self.groups):
            s += f'Group {i + 1}: {g}\n'
            s += self.estimates[g].to_string() + '\n'
        return s
        

    


def multigroup(desc, data: pd.DataFrame, group: str, mod=Model,
               **kwargs):
    """
    Perform a multi-group analysis.

    Parameters
    ----------
    desc : str or dict
        Either text description of model or an unique model for each group.
    data : pd.DataFrame
        Dataset with a group column.
    group : str
        A name of the group columns.
    mod : method, optional
        Constructor of the model. The default is Model.
    **kwargs : dict
        Extra arguments that will be passed to the fit method.

    Returns
    -------
    res : MultigroupResult
        Printable MultigroupResult.

    """
    
    groups = data[group].unique()
    res = MultigroupResult(groups, dict(), dict(), dict(), dict())
    if type(desc) is str:
        desc = {g: desc for g in groups}
    for g in groups:
        m = mod(desc[g])
        g_data = data[data[group] == g].copy()
        res.n_obs[g] = len(g_data)
        res.runs[g] = m.fit(g_data, **kwargs)
        res.estimates[g] = m.inspect()
        res.stats[g] = calc_stats(m)
    return res
