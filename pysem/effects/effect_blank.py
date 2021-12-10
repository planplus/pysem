# -*- coding: utf-8 -*-
"""This is do-nothing effect that servers entirely for demonstrational and
testing purposes"""
from .effect_base import EffectBase
import numpy as np

class EffectBlank(EffectBase):
    """
    This effect does nothing. K matrix is just a static identity matrix.
    
    The only purposes of this effect are testing and providing guidelines for
    possible developers.
    """
    def __init__(self, d_mode='identity'):
        super().__init__(None, d_mode=d_mode)

    def load(self, order, model, data, clean_start=True, **kwargs):
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
        super().load(order, model, data, clean_start, **kwargs)
        self.mx_identity = np.identity(data.shape[0])
        if clean_start:
            self.parameters = np.array([])

    def calc_k(self, model):
        return self.mx_identity

    def calc_k_grad(self, model):
        return []
