#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parser module."""
from collections import defaultdict, namedtuple
import re
from enum import Enum

Operation = namedtuple('Operation', 'name, params, onto', defaults=(None, None))
__prt_lvalue = r'(\w[\w\.]*(?:\s*,\s*[\w.]*)*)'
__prt_op = r'\s*((?:\s\w+\s)|(?:[=~\\\*@\$<>\-]+\S*?))\s*'
__prt_rvalue = r'(-?\w[\w.-]*(?:\s*\*\s*\w[\w.]*)?(?:\s*\+\s*-?\w[\w.-]*(?:\s*\*\s*\w[\w.]*)?)*)'
PTRN_EFFECT = re.compile(__prt_lvalue + __prt_op + __prt_rvalue)
PTRN_OPERATION = re.compile(r'([A-Z][A-Z_]+(?:\(.*\))?)\s*([\w\s]+)*')
PTRN_OPERATION_FULL = re.compile(r'([a-z][a-z_]*)\s*(.*?)\s*:\s*(.*)\s*')
PTRN_OPERATION_PARAM = re.compile(r'([a-z][a-z_]*)\s*[\"\'\`]\s*(.+)\s*[\"\'\`]')
PTRN_RVALUE = re.compile(r'((-?\w[\w.-]*\*)?\w[\w.]*)')
PTRN_OP = re.compile(r'(\w+)(\(.*\))?')


class SyntaxType(Enum):
    OPERATOR = 0
    OPERATION_FULL = 1
    OPERATION_PARAM = 2
    OPERATION_COMMAND = 3


def separate_token(token: str):
    """
    Test if token satisfies basic command semopy syntax and separates token.

    Parameters
    ----------
    token : str
        A text line with either effect command or operation command.

    Raises
    ------
    SyntaxError
        Token happens to be incorrect, i.e. it does not follows basic
        semopy command pattern.

    Returns
    -------
    int
        0 if effect, 1-2 if operation (depending on the format, there are 2
        formats: the first one is new small-case, the other is the old
        capital-case).
    tuple
        A tuple of (lvalue, operation, rvalue) if command is effect or
        (operation, operands) if command is operation.

    """
    effect = PTRN_EFFECT.fullmatch(token)
    if effect:
        return SyntaxType.OPERATOR, effect.groups()
    operation = PTRN_OPERATION_FULL.fullmatch(token)
    if operation:
        return SyntaxType.OPERATION_FULL, operation.groups()
    operation = PTRN_OPERATION_PARAM.fullmatch(token)
    if operation:
        return SyntaxType.OPERATION_PARAM, operation.groups()
    operation = PTRN_OPERATION.fullmatch(token)
    if operation:
        return SyntaxType.OPERATION_COMMAND, operation.groups()
    raise SyntaxError(f'Invalidate syntax for line:\n{token}')


def parse_rvalues(token: str):
    """
    Separate token by  '+' sign and parses expression "val*x" into tuples.

    Parameters
    ----------
    token : str
        Right values from operand.

    Raises
    ------
    Exception
        Raises when a certain rvalue can't be processed.

    Returns
    -------
    rvalues : dict
        A mapping Variable->Multiplicator.

    """
    token = token.replace(' ', '')
    rvalues = dict()
    for tok in token.split('+'):
        rval = PTRN_RVALUE.match(tok)
        if not rval:
            raise Exception(f'{rval} does not seem like a correct semopy expression')
        groups = rval.groups()
        name = groups[0].split('*')[-1]
        rvalues[name] = groups[1][:-1] if groups[1] else None
    return rvalues


def parse_operation(operation: str, operands: str) -> Operation:
    """
    Parse an operation according to semopy syntax.

    Parameters
    ----------
    operation : str
        Operation string with possible arguments.
    operands : str
        Variables/values that operation acts upon.

    Raises
    ------
    SyntaxError
        Rises when there is an error during parsing.

    Returns
    -------
    operation : Operation
        Named tuple containing information on operation.

    """
    oper = PTRN_OP.match(operation)
    if not oper:
        raise SyntaxError(f'Incorrect operation pattern: {operation}')
    operands = [op.strip() for op in operands.split()] if operands else list()
    groups = oper.groups()
    name = groups[0]
    params = groups[1]
    if params is not None:
        params = [t.strip() for t in params[1:-1].split(',')]
    operation = Operation(name, params, operands)
    return operation


def parse_new_operation(groups: tuple) -> Operation:
    """
    Parse an operation according to semopy syntax.

    Version for a new operation syntax.
    Parameters
    ----------
    groups : tuple
        Groups as returned by regex parser.

    Returns
    -------
    operation : Operation
        Named tuple containing information on operation.

    """
    name = groups[0]
    params = groups[1]
    try:
        try:
            operands = groups[2].split()
        except IndexError:
            operands = None
    except AttributeError:
        operands = None
        if not params:
            raise SyntaxError("Unknown syntax error.")
    operation = Operation(name, params, operands)
    return operation


def parse_desc(desc: str):
    """
    Parse a model description provided in semopy's format.

    Parameters
    ----------
    desc : str
        Model description in semopy format.

    Returns
    -------
    effects : defaultdict
        Mapping operation->lvalue->rvalue->multiplicator.
    operations : dict
        Mapping operationName->list[Operation type].

    """
    desc = desc.replace(chr(8764), chr(126))  # 替换波浪线
    effects = defaultdict(lambda: defaultdict(dict))
    operations = defaultdict(list)
    # 逐行解析
    for line in desc.splitlines():
        # 忽略注释
        try:
            i = line.index('#')
            line = line[:i]
        except ValueError:
            pass
        line = line.strip()
        if line:
            try:
                kind, items = separate_token(line)
                if kind == SyntaxType.OPERATOR:
                    lefts, op_symb, rights = items
                    rvalues = parse_rvalues(rights)
                    for left in lefts.split(','):  # 处理多个用都逗号分割的左值
                        effects[op_symb][left.strip()].update(rvalues)
                elif kind == SyntaxType.OPERATION_FULL or kind == SyntaxType.OPERATION_PARAM:
                    t = parse_new_operation(items)
                    operations[t.name].append(t)
                else:
                    operation, operands = items
                    t = parse_operation(operation, operands)
                    operations[t.name].append(t)
            except SyntaxError:
                raise SyntaxError(f"Syntax error for line:\n{line}")
    return effects, operations
