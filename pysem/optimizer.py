# -*- coding: utf-8 -*-
"""Backwards-compatibility module for semopy 1.+."""
from .model import Model
from copy import deepcopy


class Optimizer():
    def __init__(self, mod: Model):
        if not hasattr(mod, 'mx_data'):
            raise Exception('load_dataset method must be called prior to'
                            ' passing Model to an Optimizer.')
        self.model = deepcopy(mod)

    def optimize(self, obj='MLW', method='SLSQP'):
        res = self.model.fit(obj=obj, solver=method)
        return res.fun

