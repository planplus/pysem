#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base model module that contains ModelBase for lienear SEM models."""
from abc import ABC, abstractmethod
from collections import defaultdict
from .parser import parse_desc
from itertools import chain

class ModelBase(ABC):
    """
    Base model class.

    Model is a base class for linear semopy SEM models. It bears the burden of
    instantiating base SEM relatioships for their further explotation
    in practical semopy SEM models.
    """

    symb_regression = '~'
    symb_covariance = '~~'
    symb_measurement = '=~'
    symb_force_variables = 'FORCE'
    symb_define = 'DEFINE'
    symb_latent = 'latent'
    set_types = set()
    dict_effects = dict()
    dict_operations = dict()

    def __init__(self, description: str):
        """
        Instantiate base model.

        Model is a base class for linear semopy SEM models. It bears the
        of burdens instantiating base SEM relatioships for their further
        exploitation in practical semopy SEM models.
        Parameters
        ----------
        description : str
            Model description in semopy syntax.

        Returns
        -------
        None.

        """
        self.dict_effects[self.symb_regression] = self.effect_regression
        self.dict_effects[self.symb_covariance] = self.effect_covariance
        self.dict_effects[self.symb_measurement] = self.effect_measurement
        self.dict_operations[self.symb_define] = self.operation_define
        self.dict_operations[self.symb_force_variables] = self.operation_force
        self.dict_operations[self.symb_latent] = self.operation_latent
        self.description = description
        if type(description) is str:
            effects, operations = parse_desc(description)
        else:
            effects, operations = description
        self.before_classification(effects, operations)
        self.classify_variables(effects, operations)
        self.post_classification(effects)
        self.create_parameters(effects)
        self.apply_operations(operations)
        self.finalize_init()

    def before_classification(self, effects: dict, operations: dict):
        """
        Preprocess effects and operations if necessary before classification.

        Parameters
        ----------
        effects : dict
            Dict returned from parse_desc.

        operations: dict
            Dict of operations as returned from parse_desc.

        Returns
        -------
        None.

        """
        pass

    def classify_variables(self, effects: dict, operations: dict):
        """
        Classify and instantiate vars dict.

        Parameters
        ----------
        effects : dict
            Dict returned from parse_desc.

        operations: dict
            Dict of operations as returned from parse_desc.

        Returns
        -------
        None.

        """
        self.vars = dict()
        latents = set()
        in_arrows, out_arrows = set(), set()
        senders = defaultdict(set)
        indicators = set()
        for rv, inds in effects[self.symb_measurement].items():
            latents.add(rv)
            indicators.update(inds)
            out_arrows.add(rv)
            senders[rv].update(inds)
        in_arrows.update(indicators)
        for rv, lvs in effects[self.symb_regression].items():
            in_arrows.add(rv)
            out_arrows.update(lvs)
            for lv in lvs:
                senders[lv].add(rv)
        for operation in operations[self.symb_define]:
            if operation.params and operation.params[0] == 'latent':
                latents.update(operation.onto)
        for operation in operations[self.symb_latent]:
            latents.update(operation.onto)
        for operation in operations[self.symb_force_variables]:
            if operation.params and operation.params[0] == 'endo':
                in_arrows.update(operation.onto)
        allvars = out_arrows | in_arrows
        exogenous = out_arrows - in_arrows
        outputs = in_arrows - out_arrows
        endogenous = allvars - exogenous
        observed = allvars - latents
        self.vars['all'] = allvars
        self.vars['endogenous'] = endogenous
        self.vars['exogenous'] = exogenous
        self.vars['observed'] = observed
        self.vars['latent'] = latents
        self.vars['indicator'] = indicators
        self.vars['output'] = outputs
        self.vars_senders = senders

    def create_parameters(self, effects: dict):
        """
        Instantiate parameters in a model.

        Parameters
        ----------
        effects : dict
            Mapping of effects as returned by parse_desc.

        Raises
        ------
        NotImplementedError
            Raises in case of unknown effect symbol.

        Returns
        -------
        None.

        """
        for operation, items in effects.items():
            try:
                self.dict_effects[operation](items)
            except KeyError:
                raise NotImplementedError(f'{operation} is an unknown op.')

    def apply_operations(self, operations: dict):
        """
        Apply operations to model.

        Parameters
        ----------
        operations : dict
            Mapping of operations as returned by parse_desc.

        Raises
        ------
        NotImplementedError
            Raises in case of unknown command name.

        Returns
        -------
        None.

        """
        for command, items in operations.items():
            try:
                list(map(self.dict_operations[command], items))
            except KeyError:
                raise NotImplementedError(f'{command} is an unknown command.')

    def post_classification(self, effects: dict):
        """
        Procedure that is run just after classify_variables.

        Parameters
        -------
        effects : dict
            Maping opcode->values->rvalues->mutiplicator.

        Returns
        -------
        None.

        """
        pass

    def operation_force(self, operation):
        """
        Works through FORCE command.

        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Returns
        -------
        None.

        """
        pass

    def operation_define(self, operation):
        """
        Works through DEFINE command.

        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Returns
        -------
        None.

        """
        pass


    def operation_latent(self, operation):
        """
        Works through latent command.

        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Returns
        -------
        None.

        """
        pass

    @abstractmethod
    def effect_regression(self, items):
        """
        Works through regression operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        pass

    @abstractmethod
    def effect_measurement(self, items):
        """
        Works through measurement operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        pass

    @abstractmethod
    def effect_covariance(self, items):
        """
        Works through covariance operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        pass

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fits model to data.

        Returns
        -------
        None.

        """
        pass

    def finalize_init(self):
        """
        Run a post-command processing routine in the end of initialization.

        Returns
        -------
        None.

        """
        pass
