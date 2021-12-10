# -*- coding: utf-8 -*-
"""Effect with a fully parametrised K matrix"""
from ..utils import calc_zkz
from .effect_base import EffectBase
import pandas as pd
import numpy as np


class EffectFree(EffectBase):
    """
    Effect with a fully parametrised K matrix.
    
    Number of new parameters introduced is equal to g(g + 1) / 2, where g is
    an unique number of groups in data.
    """
    def __init__(self, columns: str, correlation=True, d_mode='diag'):
        """
        Instantiate EffectStatic.

        Parameters
        ----------
        columns : str
            Name of column that corresponds to individuals group id. Should
            match the appropriate row/column in the K dataframe.
        correlation : bool, optional
            If True, then K is assumed to be a correlation matrix, i.e.
            diagonal parameters are fixed to 1 and non-diagonal ones are
            constrained to (-1; 1). The default is True;
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
        self.correlation = correlation

    def load(self, i, model, data, clean_start=True, **kwargs):
        """
        Called by model new dataset is loaded.
        
        Here, Effects are configured from the data. self.parameters must be
        initialised after invoking this method.
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
        c = data[self.columns[0]]
        if clean_start:
            self.groups = list()
            for gt in c.unique():
                if type(gt) is str:
                    self.groups.extend(gt.split(';'))
                else:
                    if not np.isfinite(gt):
                        continue
                    self.groups.append(gt)
            self.groups = sorted(set(self.groups))
            p = len(self.groups)
            params = list()
            if not self.correlation:
                params.extend([0.1] * p)
            params.extend([0] * ((p * (p - 1)) // 2))
            self.num_p = p
            self.triu = np.triu_indices(p, 1)
            self.tril = np.tril_indices(p, -1)
            self.parameters = np.array(params)
        self.mx_z = calc_zkz(c, None, p_names=self.groups,
                             return_z=True)
        self.build_derivatives()

    def build_derivatives(self):
        z = self.mx_z
        grad = list()
        p = self.num_p
        if not self.correlation:
            for i in range(p):
                k = np.zeros((p, p))
                k[i, i] = 1
                grad.append(z @ k @ z.T)
        for a, b in np.nditer(self.triu):
            k = np.zeros((p, p))
            k[a, b] = 1
            k[b, a] = 1
            grad.append(z @ k @ z.T)
        self.grad = grad

    def calc_k(self, model):
        params = self.parameters
        p = self.num_p
        if not self.correlation:
            k = np.diag(params[:p])
            k[self.triu] = params[p:]
            k[self.tril] = params[p:]
        else:
            k = np.identity(p)
            k[self.triu] = params
            k[self.tril] = params
        z = self.mx_z
        return z @ k @ z.T

    def calc_k_grad(self, model):
        return self.grad

    def get_bounds(self):
        """
        Return bounding intervals for each of the effects parameters.

        Returns
        -------
        List[tuple]
            List of parameters intervals.

        """
        n = len(self.parameters)
        if self.correlation:
            return [(None, None)] * n
        p = self.num_p
        return [(0, None)] * p + [(None, None)] * (n - p)
