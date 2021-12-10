#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains Solver class that wraps around scipy and possibly other
optimization packages/procedures."""
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from copy import copy
import logging
import numpy as np


@dataclass
class SolverResult:
    """Solver result."""

    fun: float
    success: bool
    n_it: int
    x: np.ndarray
    message: str
    name_method: str
    name_obj: str = ''

    def __str__(self):
        s = list()
        if self.name_obj:
            s.append('Name of objective: ' + self.name_obj)
        s.append('Optimization method: ' + self.name_method)  
        if self.success:
            s.append('Optimization successful.')
        s.append(str(self.message))
        s.append('Objective value: {:.3f}'.format(self.fun))
        s.append('Number of iterations: {}'.format(self.n_it))
        s.append('Params: {}'.format(' '.join('{:.3f}'.format(v)
                                              for v in self.x)))
        return '\n'.join(s)

    def __bool__(self):
        return self.success


class Solver():
    """Wraps around scipy. In future, custom optimization procedures might be
    present"""

    custom_methods = dict()

    def __init__(self, method: str, fun, grad, x0, bounds=None, constrs=None,
                 options=None, **kwargs):
        """
        Instantiate Solver.

        Parameters
        ----------
        method : str
            Method name.
        fun : TYPE
            Objective function f(x).
        grad : TYPE
            Gradient function grad(x). Can be None.
        bounds: list
            List of tuples specifying bound constraints on parameters.
        constrs: dict
            Dictionary in scipy format specifying constraints.
        options : dict, optional
            Dict of options to pass to optimizers. The default is
            {'maxiter': 1000}.
        **kwargs : dict
            Extra parameters.

        Returns
        -------
        None.

        """
        if options is None:
            options = dict()
        if 'maxiter' not in options:
            options['maxiter'] = 10000
        self.method = method
        self.fun = fun
        self.grad = grad
        self.options = options
        self.extra_options = kwargs
        self.start = x0
        self.bounds = bounds
        self.constraints = constrs if constrs else list()

    def solve(self):
        """
        Solve problem.

        Returns
        -------
        SolverResult
            Information on optimization result.

        """
        if self.method in self.custom_methods:
            return self.custom_methods[self.method]()
        res = self.scipy_solve()
        if not res.success:
            logging.warning('Solver didn''t converge, see SolverResult.')
        return res

    def scipy_solve(self):
        """
        Solve problem via scipy library.

        Returns
        -------
        SolverResult
            Information on optimization result.

        """
        if self.method == 'de':
            t = copy(self.extra_options)
            if 'b_max' in t:
                bmax = t['b_max']
                del t['b_max']
            else:
                bmax = 10
            b = self.bounds
            bs = list()
            for (a, b) in b:
                if a == None:
                    a = -bmax
                if b == None:
                    b = bmax
                bs.append((a, b))
            res = differential_evolution(self.fun, bounds=bs,
                       constraints=self.constraints, **t)
        else:
            res = minimize(self.fun, self.start, jac=self.grad, bounds=self.bounds,
                           constraints=self.constraints, options=self.options,
                           method=self.method, **self.extra_options)
        try:
            nit = res.nit
        except AttributeError:
            nit = np.nan
        return SolverResult(fun=res.fun, success=res.success, n_it=nit,
                            message=res.message, x=res.x,
                            name_method=self.method)
