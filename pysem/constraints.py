#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains sympy-powered constraint parser."""
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, derive_by_array, Symbol
import numpy as np
import re

def parse_constraint(s: str, params: list):
    try:
        op = next(re.finditer('(?:>)|(?:<)|(?:=)',s))
        span = op.span()
        expr = s[:span[0]] + '-({})'.format(s[span[1]:])
        op = op.group()
        if op == '<':
            expr  = f'-({expr})'
    except StopIteration:
        raise SyntaxError('>,< or = must be present in constraint string.')
    if op in ('>', '<'):
        op = 'ineq'
    elif op == '=':
        op = 'eq'
    expr = expr.replace('^', '**')
    try:
        expr = parse_expr(expr)
    except SyntaxError:
        raise SyntaxError(f'Incorrect syntax for constraint {s}')
    params = list(map(Symbol, params))
    for v in expr.free_symbols:
        if v not in params:
            raise Exception(f"Can't add constraint: {v} is not a parameter.")
    grad = derive_by_array(expr, params)
    grad = list(map(lambda g: lambdify(params, g, 'numpy'), grad))
    grad_f = lambda x: np.array([g(*x) for g in grad])
    fun = lambdify(params, expr, 'numpy')
    f = lambda x: fun(*x)
    return {'type': op, 'fun': f, 'jac': grad_f}
