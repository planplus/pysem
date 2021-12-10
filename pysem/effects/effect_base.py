# -*- coding: utf-8 -*-
"""Abstract EffectBase class that is used as a base for all other effects."""
from abc import ABC, abstractmethod


class EffectBase(ABC):
    """
    Base effect class.
    
    This class is a part of generalized random effects model. It provides
    interface for ModelGeneralizedEffects to infer sample-wise covariance
    structure.
    """
    def __init__(self, columns=None, d_mode='diag'):
        """
        Instantiate EffectBase abstract class.

        Parameters
        ----------
        columns : tuple, str
            Column names that contain information on the data structure that
            is present in the dataset (e.g. time, region, etc).
        d_mode : str
            Mode of D matrix. If "diag", then D has unique params on the
            diagonal. If "full", then D is fully parametrised. If
            "scale", then D is an identity matrix, multiplied by a single
            variance parameter (scalar). The default is "diag".

        Returns
        -------
        None.

        """
        if type(columns) is str:
            columns = (columns, )
        elif columns is None:
            columns = list()
        self.columns = columns
        self.d_mode = d_mode
        self.parameters = None

    @abstractmethod
    def load(self, order, model, data, clean_start=True, **kwargs):
        """
        Called by model new dataset is loaded.
        
        Here, Effects are configured from the data. self.parameters must be
        initialised after invoking this method.
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
        for col in self.columns:
            if col not in data.columns:
                raise Exception(f"Effect missing necessary {col} column in"
                                " data.")
        self.order = order

    @abstractmethod
    def calc_k(self, model):
        """
        Build numpy matrix that is used as a skeleton for the covariance
        across samples matrix K.
        
        Parameters
        -------
        model : ModelGeneralizedEffects
            Instance of ModelGeneralizedEffects that calls this method.
        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        pass

    @abstractmethod
    def calc_k_grad(self, model):
        """
        Calculate gradient for matrix D.

        Parameters
        ----------
        model : ModelGeneralizedEffects
            Instance of ModelGeneralizedEffects that calls this method.
        Returns
        -------
        List[np.ndarray] of derivatives of K wrt to parameters.

        """
        pass

    def get_bounds(self):
        """
        Return bounding intervals for each of the effects parameters.

        Returns
        -------
        List[tuple]
            List of parameters intervals.

        """
        return [(None, None)] * len(self.parameters)