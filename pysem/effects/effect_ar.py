# -*- coding: utf-8 -*-
"""Autoregressive effect model."""
from .effect_base import EffectBase
import numpy as np


class EffectAR(EffectBase):
    """
    AR(1) model.
    
    At the moment, only autoregressive model of the first order is supported.
    Feel free to introduce higher order models :)
    """
    def __init__(self, columns: str, dt=1, param=None, d_mode='diag'):
        """
        Instantiate EffectAR.

        Parameters
        ----------
        columns : str
            Name of column that corresponds to time at which the individual was
            observed. Should be numeric.
        dt : float, optional
            Min difference for time to be considered different. For example,
            if dt=1, |5 - 3| / dt = 2 - lag. The default is 1.
        param : float, optional
            If provided, then AR parameter is fixed to param. The default is
            None.
        d_mode : str
            Mode of D matrix. If "diag", then D has unique params on the
            diagonal. If "full", then D is fully parametrised. If
            "scale", then D is an identity matrix, multiplied by a single
            variance parameter (scalar). The default is "diag".

        Returns
        -------
        None.

        """
        
        super().__init__(columns, d_mode=d_mode)
        self.dt = dt
        self.param = param

    def load(self, i, model, data, clean_start=True, **kwargs):
        """
        Called by model new dataset is loaded.

        Parameters
        ----------
        order : int
            Identificator of effect in model. It is just an order of the effect
            among other effects as specified by user.
        model : ModelGeneralizedEffects
            Instance of ModelGeneralizedEffects that calls this method.
        data : pd.DataFrame
            Dataset that is being loaded. Should contain self.columns.
        clean_start : bool, optional
            If True, then parameters are (re)initialized. The model will use
            the ones already present in self.parameters vector otherwise. The
            default is True.

        Returns
        -------
        None.

        """
        super().load(i, model, data, clean_start, **kwargs)
        if clean_start:
            if self.param is not None:
                self.parameters = np.array([self.param])
            else:
                self.parameters = np.array([0.0])
        g = data[self.columns[0]].values
        triu = np.triu_indices(len(g))
        d = dict()
        for i in range(len(triu[0])):
            a = triu[0][i]
            b = triu[1][i]
            dt = abs(g[a] - g[b]) // self.dt
            lt = d.get(dt, None)
            if lt is None:
                lt = [list(), list()]
                d[dt] = lt
            lt[0].append(a)
            lt[1].append(b)
        for dt, (a, b) in d.items():
            d[dt] = [np.array(a), np.array(b)]
        self.d = d
        self.num_n = len(g)
        if self.param is not None:
            t = self.param
            self.param = None
            self.mx_k = self.calc_k(model)
            self.param = t

    def calc_k(self, model):
        if self.param is not None:
            return self.mx_k
        p = self.parameters[0]
        n = self.num_n
        mx = np.zeros((n, n))
        for dt, inds in self.d.items():
            rho = p ** dt
            mx[inds] = rho
            mx[inds[::-1]] = rho
        return mx

    def calc_k_grad(self, model):
        if self.param is not None:
            return list()
        n = self.num_n
        p = self.parameters[0]
        n = self.num_n
        mx = np.zeros((n, n))
        for dt, inds in self.d.items():
            if dt == 0:
                rho = 0
            else:
                rho = dt * (p ** (dt - 1))
            mx[inds] = rho
            mx[inds[::-1]] = rho
        return mx
