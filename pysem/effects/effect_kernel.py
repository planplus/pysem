# -*- coding: utf-8 -*-
"""Effect with a pre-defied K kernel similiarity matrix."""
from sklearn.gaussian_process.kernels import Kernel
from .effect_base import EffectBase
from copy import deepcopy
import numpy as np


class EffectKernel(EffectBase):
    """
    Kernel similiarity effect.
    
    If active parameter is False, then this effect introduces no extra
    parameters other than those that come along the D matrix. Kernels are
    provided by sklearn.gaussian_process.kernels submodule.
    """
    def __init__(self, columns: str, kernel: Kernel, params: dict,
                 active=False, d_mode='diag'):
        """
        Instantiate EffectMatern.

        Parameters
        ----------
        columns : str
            Name of column that corresponds to individuals group id. Should
            match the appropriate row/column in the K dataframe.
        kernel : sklearn.gaussian_process.kernels.Kernel
            Kernel that is used to compute pairwise similarity matrix.
        params : dict
            A dictionary of parameters that are passed to kernel.
        active : bool, optional
            If True, then kernel parameters are active. Otherwise, they are
            fixed to those provided by params. The default is False.
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
        self.active = active
        if active:
            params = deepcopy(params)
            k = kernel(**params)
            ps = k.get_params()
            t = list()
            for h in k.hyperparameters:
                if h.name not in params:
                    params[h.name] = ps[h.name]
                t.append((h.name, h.n_elements))
            self.param_names = t
        self.kernel = kernel
        self.kernel_params = params

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
            if self.active:
                p = self.kernel_params
                params = list()
                for n, m in self.param_names:
                    ps = p[n]
                    if m != 1:
                        params.extend(ps)
                    else:
                        params.append(ps)
                self.parameters = np.array(params)
            else:
                self.parameters = np.array([])
        if not self.active:
            k = self.kernel(**self.kernel_params)(data[self.columns].values)
            self.mx_k = k
        else:
            self.mx_c = data[self.columns].values    

    def calc_k(self, model):
        if not self.active:
            return self.mx_k
        else:
            d = self.kernel_params
            i = 0
            p = self.parameters
            for n, m in self.param_names:
                if m == 1:
                    d[n] = p[i]
                else:
                    d[n] = p[i:m]
                i += m
            return self.kernel(**d)(self.mx_c)
    
    def calc_k_grad(self, model):
        if not self.active:
            return list()
        else:
            d = self.kernel_params
            i = 0
            p = self.parameters
            for n, m in self.param_names:
                if m == 1:
                    d[n] = p[i]
                else:
                    d[n] = p[i:m]
                i += m
            k = self.kernel(**d)(self.mx_c, eval_gradient=True)[1]
            return [k[:, :, i] for i in range(len(self.parameters))]

    def get_bounds(self):
        k = self.kernel(**self.kernel_params)
        bounds = list()
        for h in k.hyperparameters:
            for a, b in h.bounds:
                bounds.append((a, b))
        return bounds