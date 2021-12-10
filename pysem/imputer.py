#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal imputer module."""

from .model import Model
from .model_means import ModelMeans
from .model_effects import ModelEffects
from .solver import Solver
from .utils import chol_inv
from . import startingvalues
import pandas as pd
import numpy as np


class Imputer(Model):
    """Model for missing data imputation."""

    symb_missing = '@'

    matrices_names = tuple(list(Model.matrices_names) + ['data_imp'])

    def __init__(self, model: Model, data: pd.DataFrame, factors=True):
        """
        Instantiate Imputer.

        Parameters
        ----------
        model : Model
            Model.
        data : pd.DataFrame
            Data with missing data labeled as np.nan.
        factors: bool
            If True, factors are estimated. The default is True.

        Returns
        -------
        None.

        """
        self.mod = model
        self.mx_data_imp = data.copy()
        self.n_param_missing = 0
        self.factors = factors
        desc = model.description
        self.dict_effects[self.symb_missing] = self.effect_missing
        super().__init__(desc, mimic_lavaan=model.mimic_lavaan)
        self.objectives = {'ML': (self.obj_ml, self.grad_ml)}
        self.load_starting_values()

    def finalize_variable_classification(self):
        """
        Finalize variable classification.

        Reorders variables for better visual fancyness and does extra
        model-specific variable respecification.
        Returns
        -------
        None.

        """
        super().finalize_variable_classification()
        if self.factors:
            self.vars['observed'] += sorted(self.vars['latent'])

    def setup_matrices(self):
        """
        Initialize base matrix structures of the model.

        Returns
        -------
        None.

        """
        super().setup_matrices()
        for i, f in enumerate(self.start_rules):
            name = f.__name__
            if not name.endswith('_imp'):
                self.start_rules[i] = getattr(startingvalues, name + '_imp')

    def build_data_imp(self):
        """
        Build model-implied model imputed data matrix.

        Returns
        -------
        mx : np.ndarrau
            Model-implied data matrix.
        names : tuple
            Row and column names.

        """
        mx = self.mx_data_imp[self.vars['observed']].copy()
        names = (list(mx.index), list(mx.columns))
        return mx.values, names

    def preprocess_effects(self, effects: dict):
        """
        Run a routine just before effects are applied.

        Used to apply covariances to model.
        Parameters
        -------
        effects : dict
            Mapping opcode->lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        super().preprocess_effects(effects)
        for _, lvals in effects.items():
            for lval, rvals in lvals.items():
                for rval in rvals:
                    rvals[rval] = self.symb_starting_values
        missing = effects[self.symb_missing]
        obs = self.vars['observed']
        mx = self.mx_data_imp
        for v in obs:
            if v not in mx.columns:
                mx[v] = np.nan
        mx = mx[obs]
        inds = list(mx.index)
        for i, j in zip(*np.where(np.isnan(mx))):
            v = obs[j]
            missing[inds[i]][v] = None

    def effect_missing(self, items: dict):
        """
        Work through missing operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        mx, (rows, cols) = self.mx_data_imp, self.names_data_imp
        for lv, rvs in items.items():
            i = rows.index(lv)
            for rv in rvs:
                j = cols.index(rv)
                ind = (i, j)
                self.n_param_missing += 1
                name = f'_m{self.n_param_missing}'
                self.add_param(name, matrix=mx, indices=ind, start=None,
                               active=True, symmetric=False,
                               bound=(None, None))

    def fit(self, solver='SLSQP', clean_slate=False):
        """
        Perform data imputation.

        Parameters
        ----------
        solver: str
            Solver to use. The default is 'SLSQP'.

        Returns
        ----------
        SolverResult:
            Result of optimizaiton.
        """
        if clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()
        obj, grad = self.get_objective('ML')
        solver = Solver(solver, obj, grad, self.param_vals)
        res = solver.solve()
        self.update_matrices(res.x)
        return res

    def prepare_params(self):
        """
        Prepare structures for effective optimization routines.

        Returns
        -------
        None.

        """
        super().prepare_params()
        self.mx_sigma = self.calc_sigma()[0]
        try:
            self.mx_sigma_inv = chol_inv(self.mx_sigma)
        except np.linalg.LinAlgError:
            self.mx_sigma_inv = np.linalg.pinv(self.mx_sigma)

    def calc_data_grad(self):
        """
        Calculate model-implied data gradient.

        Returns
        -------
        list
            List of gradient values.

        """
        return self.mx_diffs

    def obj_ml(self, x: np.ndarray):
        """
        Calculate ML objective value.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood value.

        """
        self.update_matrices(x)
        data = self.mx_data_imp
        return np.einsum('ij,jk,ik->', data, self.mx_sigma_inv, data)

    def grad_ml(self, x: np.ndarray):
        """
        Gradient of ML objective function.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of GLS.

        """
        self.update_matrices(x)
        data_grad = self.calc_data_grad()
        t = self.mx_sigma_inv @ self.mx_data_imp.T
        return 2 * np.array([np.einsum('ij,ji->', g[4], t)
                             for g in data_grad])

    def get_fancy(self):
        """
        Returns imputed data in DataFrame form.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed data and factor scores.

        """
        data = pd.DataFrame(self.mx_data_imp, index=self.names_data_imp[0],
                            columns=self.names_data_imp[1])
        return data

    def operation_start(self, operation):
        pass

    def operation_bound(self, operation):
        pass

    def operation_constraint(self, operation):
        pass


class ImputerMeans(ModelMeans):
    """ModelMeans for missing data imputation."""

    symb_missing = '@'

    matrices_names = tuple(list(ModelMeans.matrices_names) +\
                           ['data_imp', 'g_imp'])

    def __init__(self, model: ModelMeans, data: pd.DataFrame, factors=True):
        """
        Instantiate ImputerMeans.

        Parameters
        ----------
        model : ModelMeans
            Model with meanstructure.
        data : pd.DataFrame
            Data with missing data labeled as np.nan.
        factors: bool
            If True, factors are estimated. The default is True.

        Returns
        -------
        None.

        """
        self.mod = model
        self.mx_data_imp = data.copy()
        if model.intercepts:
            data = data.copy()
            data['1'] = 1.0
        t = [v for v in model.vars['observed_exogenous']
             if v in data.columns]
        self.mx_g = data[t].copy()
        self.n_param_missing = 0
        self.factors = factors
        desc = model.description
        self.dict_effects[self.symb_missing] = self.effect_missing
        super().__init__(desc, mimic_lavaan=model.mimic_lavaan,
                         intercepts=model.intercepts)
        self.objectives = {'ML': (self.obj_ml, self.grad_ml)}
        self.load_starting_values()

    def finalize_variable_classification(self):
        """
        Finalize variable classification.

        Reorders variables for better visual fancyness and does extra
        model-specific variable respecification.
        Returns
        -------
        None.

        """
        super().finalize_variable_classification()
        if self.factors:
            self.vars['observed'] += sorted(self.vars['latent'])

    def setup_matrices(self):
        """
        Initialize base matrix structures of the model.

        Returns
        -------
        None.

        """
        super().setup_matrices()
        for i, f in enumerate(self.start_rules):
            name = f.__name__
            if not name.endswith('_imp'):
                self.start_rules[i] = getattr(startingvalues, name + '_imp')

    def build_data_imp(self):
        """
        Build model-implied model imputed data matrix.

        Returns
        -------
        mx : np.ndarray
            Model-implied data matrix.
        names : tuple
            Row and column names.

        """
        mx = self.mx_data_imp[self.vars['observed']].copy()
        names = (list(mx.columns), list(mx.index))
        return mx.values.T, names

    def build_g_imp(self):
        """
        Build model-implied model imputed data G matrix.

        Returns
        -------
        mx : np.ndarray
            Model-implied G matrix.
        names : tuple
            Row and column names.

        """
        mx = self.mx_g[self.vars['observed_exogenous']].copy()
        names = (list(mx.columns), list(mx.index))
        return mx.values.T, names

    def preprocess_effects(self, effects: dict):
        """
        Run a routine just before effects are applied.

        Used to apply covariances to model.
        Parameters
        -------
        effects : dict
            Mapping opcode->lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        super().preprocess_effects(effects)
        for _, lvals in effects.items():
            for lval, rvals in lvals.items():
                for rval in rvals:
                    rvals[rval] = self.symb_starting_values
        missing = effects[self.symb_missing]
        obs = self.vars['observed']
        mx = self.mx_data_imp
        for v in obs:
            if v not in mx.columns:
                mx[v] = np.nan
        mx = mx[obs]
        inds = list(mx.index)
        for i, j in zip(*np.where(np.isnan(mx))):
            v = obs[j]
            missing[inds[i]][v] = None

        obs = self.vars['observed_exogenous']
        mx = self.mx_g
        for v in obs:
            if v not in mx.columns:
                mx[v] = np.nan
        mx = mx[obs]
        inds = list(mx.index)
        for i, j in zip(*np.where(np.isnan(mx))):
            v = obs[j]
            missing[inds[i]][v] = None

    def effect_missing(self, items: dict):
        """
        Work through missing operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        obs_exo1 = self.vars['observed_exogenous_1']
        obs_exo2 = self.vars['observed_exogenous_2']
        for lv, rvs in items.items():
            for rv in rvs:
                if lv in obs_exo1:
                    mx = self.mx_g1
                    rows, cols = self.names_g1_imp
                elif rv in obs_exo2:
                    mx = self.mx_g2
                    rows, cols = self.names_g2_imp
                else:
                    mx, (rows, cols) = self.mx_data_imp, self.names_data_imp
                i = rows.index(rv)
                j = cols.index(lv)
                ind = (i, j)
                self.n_param_missing += 1
                name = f'_m{self.n_param_missing}'
                self.add_param(name, matrix=mx, indices=ind, start=None,
                               active=True, symmetric=False,
                               bound=(None, None))

    def fit(self, solver='SLSQP', clean_slate=False):
        """
        Perform data imputation.

        Parameters
        ----------
        solver: str
            Solver to use. The default is 'SLSQP'.

        Returns
        ----------
        SolverResult:
            Result of optimizaiton.
        """
        if not clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()
        obj, grad = self.get_objective('ML')
        solver = Solver(solver, obj, grad, self.param_vals)
        res = solver.solve()
        self.update_matrices(res.x)
        return res

    def prepare_params(self):
        """
        Prepare structures for effective optimization routines.

        Returns
        -------
        None.

        """
        super().prepare_params()
        self.mx_sigma = self.calc_sigma()[0]
        try:
            self.mx_sigma_inv = chol_inv(self.mx_sigma)
        except np.linalg.LinAlgError:
            self.mx_sigma_inv = np.linalg.pinv(self.mx_sigma)
        i = np.identity(self.mx_beta.shape[0])
        self.mx_m = self.mx_lambda @ np.linalg.inv(i - self.mx_beta)
        self.mx_g = self.mx_g_imp
        self.mx_mg = self.mx_m @ self.mx_gamma1 + self.mx_gamma2

    def calc_data_grad(self):
        """
        Calculate model-implied data gradient.

        Returns
        -------
        list
            List of gradient values.

        """
        grad = list()
        for mxs in self.mx_diffs:
            g = np.float32(0.0)
            if mxs[-2] is not None:  # data_imp
                g += mxs[-2]
            if mxs[-1] is not None:  # g1
                g -= self.mx_mg @ mxs[-1]
            grad.append(g)
        return grad

    def obj_ml(self, x: np.ndarray):
        """
        Calculate ML objective value.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood value.

        """
        self.update_matrices(x)
        center = self.mx_data_imp - self.calc_mean(self.mx_m)
        return np.einsum('ji,jk,ki->', center, self.mx_sigma_inv, center)

    def grad_ml(self, x: np.ndarray):
        """
        Gradient of ML objective function.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of ML.

        """
        self.update_matrices(x)
        data_grad = self.calc_data_grad()
        center = self.mx_data_imp - self.calc_mean(self.mx_m)
        t = self.mx_sigma_inv @ center
        return 2 * np.array([np.einsum('ji,ji->', g, t)
                             for g in data_grad])

    def get_fancy(self):
        """
        Returns imputed data in DataFrame form.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed data and factor scores.

        """
        data = pd.DataFrame(self.mx_data_imp, index=self.names_data_imp[0],
                            columns=self.names_data_imp[1])
        g = pd.DataFrame(self.mx_g_imp, index=self.names_g_imp[0],
                            columns=self.names_g_imp[1])
        if '1' in g.index:
            g.drop('1', inplace=True)
        res = pd.concat([data, g]).T
        var = self.mod.vars
        t = sorted(var['all'] - var['latent']) + sorted(var['latent'])
        return res[t]

    def operation_start(self, operation):
        pass

    def operation_bound(self, operation):
        pass

    def operation_constraint(self, operation):
        pass

class ImputerEffects(ModelEffects):
    """ModelEffects for missing data imputation."""

    symb_missing = '@'

    matrices_names = tuple(list(ModelEffects.matrices_names) +\
                           ['data_imp', 'g_imp'])

    def __init__(self, model: ModelMeans, data: pd.DataFrame, group: str, 
                 k=None, factors=True):
        """
        Instantiate ImputerMeans.

        Parameters
        ----------
        model : ModelMeans
            Model with meanstructure.
        data : pd.DataFrame
            Data with missing data labeled as np.nan.
        k : pd.DataFrame
            Kinship between individuals matrix. If None, identity is assumed.
            The default is None.
        factors: bool
            If True, factors are estimated. The default is True.

        Returns
        -------
        None.

        """
        self.mod = model
        if model.intercepts:
            data = data.copy()
            data['1'] = 1.0
        self.mx_data_orig = data.copy()
        self.mx_data_imp = data
        t = [v for v in model.vars['observed_exogenous']
             if v in data.columns]
        self.mx_g = data[t].copy()
        self.n_param_missing = 0
        self.factors = factors
        desc = model.description
        self.dict_effects[self.symb_missing] = self.effect_missing
        super().__init__(desc, mimic_lavaan=model.mimic_lavaan,
                         intercepts=model.intercepts)
        self.objectives = {'ML': (self.obj_ml, None)}
        self.load_starting_values()
        self.load(group, k=k)

    def load(self, group: str, k=None):
        """
        Load kinship K matrix.

        Parameters
        ----------
        group : str
            Name of column with group labels.
        k : pd.DataFrame
            Covariance matrix across rows, i.e. kinship matrix. If None,
            identity is assumed. The default is None.

        KeyError
            Rises when there are missing variables from the data.
        Exception
            Rises when group parameter is None.
        Returns
        -------
        None.

        """
        obs = self.vars['observed']
        data = self.mx_data_orig
        grs = data[group]
        p_names = list(grs.unique())
        p, n = len(p_names), data.shape[0]
        if k is None:
            k = np.identity(p)
        elif k.shape[0] != p:
            raise Exception("Dimensions of K don't match number of groups.")
        z = np.zeros((n, p))
        for i, germ in enumerate(grs):
            j = p_names.index(germ)
            z[i, j] = 1.0
        if type(k) is pd.DataFrame:
            try:
                k = k.loc[p_names, p_names].values
            except KeyError:
                raise KeyError("Certain groups in K differ from those "\
                                "provided in a dataset.")
        self.mx_g = data[self.vars['observed_exogenous']].values.T
        if len(self.mx_g.shape) != 2:
            self.mx_g = self.mx_g[np.newaxis, :]
        g = self.mx_g
        self.num_m = self.vars['observed']
        zkz = z @ k @ z.T
        c = np.linalg.inv(np.identity(self.mx_beta.shape[0]) - self.mx_beta)
        m = self.mx_lambda @ c
        sigma = m @ self.mx_psi @ m.T + self.mx_theta
        self.mx_m = m
        m = len(obs)
        r = n * (np.ones((m, m)) * self.mx_v[0, 0] + sigma)\
            + np.trace(zkz) * self.mx_d
        w = np.ones((n, n)) * np.trace(sigma) + zkz * np.trace(self.mx_d) +\
            np.identity(n) * self.mx_v[0, 0] * m
        self.mx_r_inv = chol_inv(r)
        self.mx_w_inv = chol_inv(w)

    def finalize_variable_classification(self):
        """
        Finalize variable classification.

        Reorders variables for better visual fancyness and does extra
        model-specific variable respecification.
        Returns
        -------
        None.

        """
        super().finalize_variable_classification()
        if self.factors:
            self.vars['observed'] += sorted(self.vars['latent'])

    def setup_matrices(self):
        """
        Initialize base matrix structures of the model.

        Returns
        -------
        None.

        """
        super().setup_matrices()
        for i, f in enumerate(self.start_rules):
            name = f.__name__
            if not name.endswith('_imp'):
                self.start_rules[i] = getattr(startingvalues, name + '_imp')

    def build_data_imp(self):
        """
        Build model-implied model imputed data matrix.

        Returns
        -------
        mx : np.ndarray
            Model-implied data matrix.
        names : tuple
            Row and column names.

        """
        mx = self.mx_data_imp[self.vars['observed']].copy()
        names = (list(mx.columns), list(mx.index))
        return mx.values.T, names

    def build_g_imp(self):
        """
        Build model-implied model imputed data G matrix.

        Returns
        -------
        mx : np.ndarray
            Model-implied G matrix.
        names : tuple
            Row and column names.

        """
        mx = self.mx_g[self.vars['observed_exogenous']].copy()
        names = (list(mx.columns), list(mx.index))
        return mx.values.T, names

    def preprocess_effects(self, effects: dict):
        """
        Run a routine just before effects are applied.

        Used to apply covariances to model.
        Parameters
        -------
        effects : dict
            Mapping opcode->lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        super().preprocess_effects(effects)
        for _, lvals in effects.items():
            for lval, rvals in lvals.items():
                for rval in rvals:
                    rvals[rval] = self.symb_starting_values
        missing = effects[self.symb_missing]
        obs = self.vars['observed']
        mx = self.mx_data_imp
        for v in obs:
            if v not in mx.columns:
                mx[v] = np.nan
        mx = mx[obs]
        inds = list(mx.index)
        for i, j in zip(*np.where(np.isnan(mx))):
            v = obs[j]
            missing[inds[i]][v] = None

        obs = self.vars['observed_exogenous']
        mx = self.mx_g
        for v in obs:
            if v not in mx.columns:
                mx[v] = np.nan
        mx = mx[obs]
        inds = list(mx.index)
        for i, j in zip(*np.where(np.isnan(mx))):
            v = obs[j]
            missing[inds[i]][v] = None

    def effect_missing(self, items: dict):
        """
        Work through missing operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        obs_exo1 = self.vars['observed_exogenous_1']
        obs_exo2 = self.vars['observed_exogenous_2']
        for lv, rvs in items.items():
            for rv in rvs:
                if lv in obs_exo1:
                    mx = self.mx_g1
                    rows, cols = self.names_g1_imp
                elif rv in obs_exo2:
                    mx = self.mx_g2
                    rows, cols = self.names_g2_imp
                else:
                    mx, (rows, cols) = self.mx_data_imp, self.names_data_imp
                i = rows.index(rv)
                j = cols.index(lv)
                ind = (i, j)
                self.n_param_missing += 1
                name = f'_m{self.n_param_missing}'
                self.add_param(name, matrix=mx, indices=ind, start=None,
                               active=True, symmetric=False,
                               bound=(None, None))

    def fit(self, solver='SLSQP', clean_slate=False):
        """
        Perform data imputation.

        Parameters
        ----------
        solver: str
            Solver to use. The default is 'SLSQP'.

        Returns
        ----------
        SolverResult:
            Result of optimizaiton.
        """
        if clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()
        obj, grad = self.get_objective('ML')
        solver = Solver(solver, obj, grad, self.param_vals)
        res = solver.solve()
        self.update_matrices(res.x)
        return res

    def prepare_params(self):
        """
        Prepare structures for effective optimization routines.

        Returns
        -------
        None.

        """
        super().prepare_params()

    def calc_data_grad(self):
        """
        Calculate model-implied data gradient.

        Returns
        -------
        list
            List of gradient values.

        """
        grad = list()
        for mxs in self.mx_diffs:
            g = np.float32(0.0)
            if mxs[-2] is not None:  # data_imp
                g += mxs[-2]
            if mxs[-1] is not None:  # g1
                g -= self.mx_mg @ mxs[-1]
            grad.append(g)
        return grad

    def obj_ml(self, x: np.ndarray):
        """
        Calculate ML objective value.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood value.

        """
        self.update_matrices(x)
        center = self.mx_data_imp - self.calc_mean(self.mx_m)
        r, w = self.mx_r_inv, self.mx_w_inv
        return np.einsum('ij,jk->', w @ center.T, r @ center)

    def grad_ml(self, x: np.ndarray):
        """
        Gradient of ML objective function.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of ML.

        """
        self.update_matrices(x)
        data_grad = self.calc_data_grad()
        center = self.mx_data_imp - self.calc_mean(self.mx_m)
        t = self.mx_sigma_inv @ center
        return 2 * np.array([np.einsum('ji,ji->', g, t)
                             for g in data_grad])

    def get_fancy(self):
        """
        Returns imputed data in DataFrame form.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed data and factor scores.

        """
        data = pd.DataFrame(self.mx_data_imp, index=self.names_data_imp[0],
                            columns=self.names_data_imp[1])
        g = pd.DataFrame(self.mx_g_imp, index=self.names_g_imp[0],
                            columns=self.names_g_imp[1])
        if '1' in g.index:
            g.drop('1', inplace=True)
        res = pd.concat([data, g]).T
        var = self.mod.vars
        t = sorted(var['all'] - var['latent']) + sorted(var['latent'])
        return res[t]

    def operation_start(self, operation):
        pass

    def operation_bound(self, operation):
        pass

    def operation_constraint(self, operation):
        pass


def get_imputer(self):
    """
    Retrieve an appropriate Imputer instance.

    Parameters
    ----------
    self : Model or ModelMeans
        Model.

    Returns
    -------
    Imputer, ImputerMeans or ImputerEffects.

    """
    if type(self) is Model:
        return Imputer
    elif type(self) is ModelMeans:
        return ImputerMeans
    elif type(self) is ModelEffects:
        return ImputerEffects
