"""Different utility functions for internal usage."""
import scipy.linalg.lapack as lapack
import pandas as pd
import numpy as np


def duplication_matrix(n: int):
    """
    Create duplication matrix D such that D vech(X) = vec(X).

    Parameters
    ----------
    n : int
        Size n of n x n matrix X.

    Returns
    -------
    Duplication matrix D.

    """
    m = n * n
    k = (n + 1) * n // 2
    d = np.zeros((m, k))
    for i in range(n):
        for j in range(i + 1):
            a1 = i * n + j
            a2 = j * n + i
            b = j * n + i - j * (j + 1) // 2
            d[a1, b] = 1
            d[a2, b] = 1
    return d


def kron_identity(mx: np.ndarray, sz: int, back=False):
    """
    Calculate Kronecker product with identity matrix.

    Simulates np.kron(mx, np.identity(sz)).
    Parameters
    ----------
    mx : np.ndarray
        Matrix.
    sz : int
        Size of identity matrix.
    back : bool, optional
        If True, np.kron(np.identity(sz), mx) will be calculated instead. The
        default is False.

    Returns
    -------
    np.ndarray
        Kronecker product of mx and an indeity matrix.

    """
    m, n = mx.shape
    r = np.arange(sz)
    if back:
        out = np.zeros((sz, m, sz, n), dtype=mx.dtype)
        out[r, :, r, :] = mx
    else:
        out = np.zeros((m, sz, n, sz), dtype=mx.dtype)
        out[:, r, :, r] = mx
    out.shape = (m * sz, n * sz)
    return out


def delete_mx(mx: np.ndarray, exclude: np.ndarray):
    """
    Remove column and rows from square matrix.

    Parameters
    ----------
    mx : np.ndarray
        Square matrix.
    exclude : np.ndarray
        List of indices corresponding to rows/cols.

    Returns
    -------
    np.ndarray
        Square matrix without certain rows and columns.

    """
    return np.delete(np.delete(mx, exclude, axis=0), exclude, axis=1)


def cov(x: np.ndarray):
    """
    Compute covariance matrix takin in account missing values.

    Parameters
    ----------
    x : np.ndarray
        Data.

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """
    masked_x = np.ma.array(x, mask=np.isnan(x))
    cov = np.ma.cov(masked_x, bias=True, rowvar=False).data
    if cov.size == 1:
        cov.resize((1, 1))
    return cov


def cor(x: np.ndarray):
    """
    Compute correlation matrix takin in account missing values.

    Parameters
    ----------
    x : np.ndarray
        Data.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """
    masked_x = np.ma.array(x, mask=np.isnan(x))
    cor = np.ma.corrcoef(masked_x, bias=True, rowvar=False).data
    if cor.size == 1:
        cor.resize((1, 1))
    return cor


def chol(x: np.array, inv=True):
    """
    Calculate cholesky decomposition of matrix.

    Parameters
    ----------
    x : np.ndarray
        Matrix.
    inv : bool, optional
        If True, returns L^{-1} instead. The default i True.

    Returns
    -------
    Cholesky decomposition matrix L.

    """
    c, info = lapack.dpotrf(x)
    if inv:
        lapack.dtrtri(c, overwrite_c=1)
    return c


def chol_inv(x: np.array):
    """
    Calculate invserse of matrix using Cholesky decomposition.

    Parameters
    ----------
    x : np.array
        Data with columns as variables and rows as observations.

    Raises
    ------
    np.linalg.LinAlgError
        Rises when matrix is either ill-posed or not PD.

    Returns
    -------
    c : np.ndarray
        x^(-1).

    """
    c, info = lapack.dpotrf(x)
    if info:
        raise np.linalg.LinAlgError
    lapack.dpotri(c, overwrite_c=1)
    return c + c.T - np.diag(c.diagonal())


def chol_inv2(x: np.ndarray):
    """
    Calculate invserse and logdet of matrix using Cholesky decomposition.

    Parameters
    ----------
    x : np.ndarray
        Data with columns as variables and rows as observations.

    Raises
    ------
    np.linalg.LinAlgError
        Rises when matrix is either ill-posed or not PD.

    Returns
    -------
    c : np.ndarray
        x^(-1).
    logdet : float
        ln|x|

    """
    c, info = lapack.dpotrf(x)
    if info:
        raise np.linalg.LinAlgError
    d = c.diagonal()
    logdet = 2 * np.sum(np.log(d))
    lapack.dpotri(c, overwrite_c=1)
    return c + c.T - np.diag(d), logdet


def compare_results(model, true: pd.DataFrame, error='relative',
                    ignore_cov=True, drop_equal=True, mandatory=False,
                    return_table=False):
    """
    Compare parameter estimates in model to parameter values in a DataFrame.

    Parameters
    ----------
    model : Model
        Model instance.
    true : pd.DataFrame
        DataFrame with operations and expected estimates. Should have "lval",
        "op", "rval", "Value" columns in this particular order.
    error : str, optional
        If 'relative', relative errors are calculated. Absolute errors are
        calculated otherwise. The default is 'relative'.
    ignore_cov : bool, optional
        If True, then covariances (~~) are ignored. The default is True.
    drop_equal : bool, optional
        If True, then parameters that are exactly equal in model are dropped.
        Effectively, it clears result from fixed loadings in Lambda. The
        default is True.
    mandatory : bool, optional
        If True, then then all of entries in true DataFrame must have their
        counterparts in model. It might be a problem if there is a known
        variance estimate for an exogenous variable and you are testing
        ModelMeans/ModelEffects that lack variance parameter. The default is
        False.
    return_table : bool, optional
        If True, then pd.DataFrame table is returned instead of nplist.

    Raises
    ------
    Exception
        Rise when operation present in true is not present in the model.

    Returns
    -------
    errs : list
        List of errors.

    """
    if type(model) is not pd.DataFrame:
        ins = model.inspect(information=None)
    else:
        ins = model
    errs = list()
    if return_table:
        ops = list()
        lvals = list()
        rvals = list()
        ests = list()
        trues = list()
    for row in true.iterrows():
        lval, op, rval, value = row[1].values[:4]
        if op == '~~' and ignore_cov:
            continue
        if op == '=~':
            op = '~'
            lval, rval = rval, lval
        est = ins[(ins.lval == lval) & (ins.op == op) & (ins.rval == rval)]
        if len(est) == 0:
            if mandatory:
                raise Exception(f'Unknown estimate: {row}.')
            continue
        est = est.Estimate.values[0]
        if drop_equal and est == value:
            continue
        if error == 'relative':
            errs.append(abs((value - est) / est))
        else:
            errs.append(abs(value - est))
        if return_table:
            ops.append(op)
            lvals.append(lval)
            rvals.append(rval)
            ests.append(est)
            trues.append(value)
    if return_table:
        d = {'lval': lvals, 'op': ops, 'rval': rvals, 'Error': errs,
             'Estimate': ests, 'True': trues}
        df = pd.DataFrame.from_dict(d).sort_values('Error', axis=0,
                                                   ascending=False)
        return df
    return errs


def calc_zkz(groups: pd.Series, k: pd.DataFrame, p_names=None,
             return_z=False):
    """
    Calculate ZKZ^T relationship matrix from covariance matrix K.

    Parameters
    ----------
    groups : pd.Series
        Series of group names for individuals.
    k : pd.DataFrame
        Covariance-across-groups matrix. If None, then its calculate as an
        identity matrix.
    p_names : List[str], optional
        List of unique group names. If False, then it is determined from the
        groups variable. The default is None.
    return_z : bool, optional
        If True, then only Z is returned. The default is False.

    Raises
    ------
    KeyError
        Incorrect group naming.

    Returns
    -------
    np.ndarray
        ZKZ^T matrix.

    """
    if p_names is None:
        lt = list()
        p_names = groups.unique()
        for gt in groups.unique():
            if type(gt) is str:
                lt.extend(gt.split(';'))
            else:
                if not np.isfinite(gt):
                    continue
                lt.append(gt)
        p_names = sorted(set(lt))
    p, n = len(p_names), len(groups)
    z = np.zeros((n, p))
    for i, germ in enumerate(groups):
        if type(germ) is str:
            for g in germ.split(';'):
                try:
                    j = p_names.index(g)
                    z[i, j] = 1.0
                except ValueError:
                    continue
        else:
            try:
                j = p_names.index(germ)
                z[i, j] = 1.0
            except ValueError:
                continue
    if return_z:
        return z
    if k is None:
        k = np.identity(p)
    if type(k) is pd.DataFrame:
        try:
            k = k.loc[p_names, p_names].values
        except KeyError:
            raise KeyError("Certain groups in K differ from those "
                           "provided in a dataset.")
    zkz = z @ k @ z.T
    return zkz


def calc_reduced_ml(model, variables: set, x=None,
                    exclude=False):
    """
    Calculate ML with restricted/reduced sigma.

    Parameters
    ----------
    model : Model
        DESCRIPTION.
    variables : set
        List/set of variable to be included in Sigma.
    x : np.ndarray, optional
        Paremters vector. The default is None.
    exclude : bool, optional
        If True, then variables are a list to be excluded, not included. The
        default is False.

    Returns
    -------
    float
        Degenerate ML.

    """

    from .model import Model
    if type(model) is not Model:
        raise Exception('ModelMeans or ModelEffects degenerate ML is not '
                        'supported.')
    obs = model.vars['observed']
    if exclude:
        inds = [obs.index(v) for v in variables]
    else:
        inds = [i for i, v in enumerate(obs) if v not in variables]

    def deg_sigma():
        sigma, (m, c) = true_sigma()
        sigma = delete_mx(sigma, inds)
        m = np.delete(m, inds, axis=0)
        return sigma, (m, c)

    true_sigma = model.calc_sigma
    true_cov = model.mx_cov
    true_covlogdet = model.cov_logdet
    model.mx_cov = delete_mx(true_cov, inds)
    model.cov_logdet = np.linalg.slogdet(model.mx_cov)[1]
    model.calc_sigma = deg_sigma
    if x is None:
        x = model.param_vals
    if type(model) is Model:
        ret = model.obj_mlw(x)
    else:
        ret = model.obj_fiml(x)
    model.calc_sigma = true_sigma
    model.mx_cov = true_cov
    model.cov_logdet = true_covlogdet
    return ret
