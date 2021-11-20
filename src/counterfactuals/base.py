"""Base classes for all counterfactual generation models"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class CounterfactualMethod(ABC):

    @abstractmethod
    def generate(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Generate counterfactuals for a single example x
        :param x: array-like of shape (n_features,)
            single example to generate counterfactuals for
        :return: array-like of shape (n_counterfactuals, n_features)
            generated counterfactuals
        """
        pass
