# -*- coding: utf-8 -*-
"""Moving average effect model."""
from .effect_base import EffectBase
import numpy as np


class EffectMA(EffectBase):
    """
    MA(1) and MA(2) models.
    
    Number of new parameters introduced is equal to p in M(p).
    """
    def __init__(self, columns: str, order=1,
                 dt_bounds=None,
                 d_mode='diag'):
        """
        Instantiate EffectMA.

        Parameters
        ----------
        columns : str
            Name of column that corresponds to time at which the individual was
            observed. Should be numeric.
        order : int, optional
            Order "p" of the MA(p) model. The default is 1.
        dt_bounds : Tuple[tuple], optional
            List of tuples that of length p (order) that contain bounding boxes
            for autocorrelation function AF(dt=|t_i - t_j|). For instance,
            if p = 2 and dt_bounds = ((0, 1), (1, 4), (4,  10)), then AF for
            elements observed at times 121 and 118 is AF(1) as
            0 <= (121 - 118) = 3 < 4;  AF for elements observed at times 121
            and 300 is 0 as it is beyond any of the bounding boxes specified.
            If None, then ((0, 1), (1, 2), (2, 3)...(p, p + 1)) is assumed.
            The default is None.
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
        self.ma_order = order
        if dt_bounds is None:
            self.dt_bounds = [(i, i + 1) for i in range(order + 1)]
        else:
            self.dt_bounds = dt_bounds

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
            self.parameters = np.array([0.0] * self.ma_order)
        g = data[self.columns[0]].values
        inds_a = [list() for _ in range(self.ma_order + 1)]
        inds_b = [list() for _ in range(self.ma_order + 1)]
        triu = np.triu_indices(len(g))
        for i in range(len(triu[0])):
            a = triu[0][i]
            b = triu[1][i]
            dt = abs(g[a] - g[b])
            for i, (c, d) in enumerate(self.dt_bounds):
                if c <= dt < d:
                    inds_a[i].append(a)
                    inds_b[i].append(b)
                    break
        self.inds = [(np.array(a), np.array(b)) for a, b in zip(inds_a,
                                                                inds_b)]
        self.num_n = len(g)


    def calc_k(self, model):
        params = self.parameters
        n = self.num_n
        mx = np.zeros((n, n))
        denom = 1 + (params ** 2).sum()
        for i, ind in enumerate(self.inds):
            if not len(ind[0]):
                continue
            if i == 0:
                num = 1
            else:
                num = params[i - 1]
            for p in range(1, self.ma_order - i + 1):
                num += params[p - 1] * params[p + i - 1]
            rho = num / denom
            mx[ind] = rho
            mx[ind[::-1]] = rho
        return mx
            

    def calc_k_grad(self, model):
        params = self.parameters
        n = self.num_n
        m = len(params)
        grad = [np.zeros((n, n)) for _ in range(m)]
        mx = np.zeros((n, n))
        denom = 1 + (params ** 2).sum()
        for i, ind in enumerate(self.inds):
            dro = np.array([0.0] * m)
            if i == 0:
                num = 1
            else:
                num = params[i - 1]
                dro[i - 1] += 1
            for p in range(1, self.ma_order - i + 1):
                num += params[p - 1] * params[p + i - 1]
                dro[p - 1] += params[p + i - 1] 
                dro[p + i - 1] += params[p - 1] 
            dro /= denom
            d1 = - 2 * num / (denom ** 2)
            for p in range(m):
                dro[p] += d1 * params[p]
                mx = grad[p]
                mx[ind] += dro[p]
                mx[ind[::-1]] += dro[p]
        return grad
