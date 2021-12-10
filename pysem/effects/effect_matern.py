# -*- coding: utf-8 -*-
"""Effect with a K matern covariance/similiarity matrix."""
from sklearn.gaussian_process.kernels import Matern
from .effect_kernel import EffectKernel


class EffectMatern(EffectKernel):
    """
    Matern covariance matrices are useful in spatial analysis.
    
    If 'active' parameter is False, then this effect introduces no extra
    parameters other than those that come along the D matrix.
    """
    def __init__(self, columns: str, nu=float('inf'), rho=1.0, 
                 active=False, d_mode='diag'):
        """
        Instantiate EffectMatern.

        Parameters
        ----------
        columns : str
            Name of column that corresponds to individuals group id. Should
            match the appropriate row/column in the K dataframe.
        nu : float, optional
            Num parameter of the matern kernel. The default is inf.
        rho : float, optional
            Rho parameter of the matern kernel, i.e. length. The default is 1.
        active : bool, optional
            If True, then rho is an active parameter that is optimized.
            Otherwise, it is fixed. The default is False.
        d_mode : str
            Mode of D matrix. If "diag", then D has unique params on the
            diagonal. If "full", then D is fully parametrised. If
            "scale", then D is an identity matrix, multiplied by a single
            variance parameter (scalar). The default is "diag".

        Returns
        -------
        None.

        """
        p = {'length_scale': rho,
             'nu': nu}
        super().__init__(columns, kernel=Matern, params=p, active=active,
                         d_mode=d_mode)
