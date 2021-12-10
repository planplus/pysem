# -*- coding: utf-8 -*-
"""Effect with a pre-defied K matrix as in ModelEffects."""
from ..utils import calc_zkz
from .effect_base import EffectBase
import pandas as pd
import numpy as np


class EffectStatic(EffectBase):
    """
    Effect with a pre-defined static K matrix as in classical LMMs.
    
    This effect introduces no extra parameters other than those that come along
    the D matrix.
    """
    def __init__(self, columns: str, k: pd.DataFrame, d_mode='diag'):
        """
        Instantiate EffectStatic.

        Parameters
        ----------
        columns : str
            Name of column that corresponds to individuals group id. Should
            match the appropriate row/column in the K dataframe.
        k : pd.DataFrame
            K matrix with columns and rows subscribed with accorance to the
            existing "groups" that encapsulate individual observations.
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
        self.k = k

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
        if clean_start:
            self.parameters = np.array([])
        mx_k = kwargs.get(f'k_{self.order+1}', self.k)
        c = data[self.columns[0]]
        self.mx_k = calc_zkz(c, mx_k)

    def calc_k(self, model):
        return self.mx_k

    def calc_k_grad(self, model):
        return []