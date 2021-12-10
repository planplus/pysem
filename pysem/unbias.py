# -*- coding: utf-8 -*-
"""Second order de-biasing correction"""
from .model_generation import generate_data
from copy import deepcopy
import pandas as pd
import numpy as np

def bias_correction(model, n=100, resample_mean=False, extra_data=None,
                    max_rel_fun=1000, clean_slate=False, **kwargs):
    """
    Parametric Bootstrap bias correction.

    Parameters
    ----------
    model : TYPE
        Model or ModelMeans. Tehnically, ModelEffects and
        ModelGeneralizedEffects will work too, but the results are almost
        guaranteed to be bad. It will probably change eventually, but not
        anytime soon.
    n : int, optional
        Number of bootstrap iterations. The default is 100.
    resample_mean : bool, optional
        If True, then mean components in ModelMeans are also resampled. The
        default is False.
    extra_data : pd.DataFrame, optional
        Extra dataframe to append to resampled data. The default is None.
    max_rel_fun : float, optional
        Bootstrap iteration is skipped if optimizer failed or if the resulting
        loss function is greater than max_rel_fun times original MLE loss fun.
        The default is 3.
    clean_slate : bool, optional
        If True, then each PBS iteration uses previous values as starting ones.
        May speed-up computations. The default is True.
    **kwargs : dict
        Extra arguments to be passed to fit method.

    Returns
    -------
    None.

    """
    
    model_c = deepcopy(model)
    t = None
    try:
        n_samples = model.n_samples
    except AttributeError:
        n_samples = model.mx_data[0]
    if extra_data is not None:
        extra_cols = set(extra_data.columns)
    i = 0
    row_fails = 0
    px = model.param_vals.copy()
    fun = model.last_result.fun
    while i < n:
        if not resample_mean:
            data = generate_data(model, n=n_samples)
        else:
            data = generate_data(model, n=n_samples, generator_exo=None)
        if extra_data is not None:
            data.index = extra_data.index
            st = extra_cols - set(data.columns)
            data = pd.concat([data, extra_data[st]], axis=1)
        cov = data.cov()
        r = model_c.fit(data, cov=cov, clean_slate=clean_slate, **kwargs)
        if not r.success or (r.fun > max_rel_fun * fun):
            row_fails += 1
            if row_fails == 40:
                raise np.linalg.LinAlgError("Couldn't sample proper data.")
            continue
        if t is None:
            t = r.x
        else:
            t += r.x
        model.param_vals[:] = px
        i += 1
        row_fails = 0
    model.param_vals = 2 * px - t / n
    
